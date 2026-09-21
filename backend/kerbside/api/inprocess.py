"""Call the API in-process, with no HTTP server underneath it.

This is what lets the console run on a static host. In the browser build the
whole ``kerbside`` package runs inside the page under Pyodide (CPython compiled
to WebAssembly), and the frontend's requests are handed to :func:`dispatch`
instead of going over the network. The pipeline doing the work is the same code
the server runs - not a port, not a JavaScript reimplementation.

It dispatches to the **real route functions** in :mod:`kerbside.api.routes`,
resolved from FastAPI's own route table: path, query and body parameters are
read from each route's declared signature and validated by the same Pydantic
models. So a route added to the server is available here with no change, and a
request that fails validation on the server fails the same way in the browser.

What it deliberately does not do is run FastAPI's ASGI machinery. That machinery
hands synchronous endpoints to a thread pool, and WebAssembly in a browser has
no threads; calling the endpoint function directly sidesteps the only part that
cannot work there. Two transports have no in-process meaning and are not
reachable: the MJPEG stream (the page advances the cursor itself and asks for
one frame at a time) and the telemetry WebSocket (``GET /api/telemetry`` returns
the same payload on request).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any
from urllib.parse import parse_qsl

import starlette.responses
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute
from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.responses import Response, StreamingResponse

__all__ = ["dispatch", "dispatch_text", "set_progress"]

#: The transports that only make sense over a real connection.
_UNREACHABLE = {"/api/stream.mjpg"}


async def _iterate_inline(iterator: Iterable[Any]) -> AsyncIterator[Any]:
    for chunk in iterator:
        yield chunk


# Starlette wraps a synchronous body iterator in ``iterate_in_threadpool``,
# which needs a worker thread. Iterating inline is exactly equivalent for a
# body that is already fully computed, which is all this API ever streams.
starlette.responses.iterate_in_threadpool = _iterate_inline  # type: ignore[assignment]


class _Request:
    """The one part of ``starlette.requests.Request`` the routes read."""

    def __init__(self, query: dict[str, str]) -> None:
        self.query_params = query


def _routes() -> list[APIRoute]:
    # Imported lazily so that importing this module does not pull in the whole
    # pipeline before the caller has had a chance to install a progress hook.
    from kerbside.api import routes

    return [r for r in routes.router.routes if isinstance(r, APIRoute)]


def _coerce(field: Any, raw: Any) -> Any:
    # Lax-mode validation turns "95" into 95 and "true" into True, the same
    # conversion FastAPI applies to path and query strings.
    return TypeAdapter(field.field_info.annotation).validate_python(raw)


def _json(status: int, payload: Any) -> tuple[int, str, bytes]:
    return status, "application/json", json.dumps(jsonable_encoder(payload)).encode("utf-8")


async def _body_of(response: Response) -> bytes:
    if isinstance(response, StreamingResponse):
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk))
        return b"".join(chunks)
    return bytes(response.body)


async def dispatch(
    method: str,
    path: str,
    query: dict[str, str] | None = None,
    body: Any = None,
) -> tuple[int, str, bytes]:
    """Handle one request; return ``(status, content_type, body_bytes)``.

    Errors come back as the server would send them: a 400 with the route's
    message, a 422 for a body that fails validation, a 404 for an unknown path.
    A 500 carries the exception's type and message rather than nothing, because
    in the browser there is no server log to go and read.
    """
    method = method.upper()
    query = dict(query or {})

    if path in _UNREACHABLE:
        return _json(
            404,
            {"detail": f"{path} needs a live connection and is not available in-process"},
        )

    for route in _routes():
        if method not in route.methods:
            continue
        match = route.path_regex.match(path)
        if match is None:
            continue
        dependant = route.dependant
        kwargs: dict[str, Any] = {}
        try:
            for field in dependant.path_params:
                kwargs[field.name] = _coerce(field, match.group(field.alias or field.name))
            for field in dependant.query_params:
                key = field.alias or field.name
                if key in query:
                    kwargs[field.name] = _coerce(field, query[key])
                elif field.field_info.is_required():
                    return _json(422, {"detail": f"missing query parameter {key!r}"})
                else:
                    kwargs[field.name] = field.get_default()
            for field in dependant.body_params:
                model = field.field_info.annotation
                payload = body if body is not None else {}
                if isinstance(model, type) and issubclass(model, BaseModel):
                    kwargs[field.name] = model.model_validate(payload)
                else:
                    kwargs[field.name] = TypeAdapter(model).validate_python(payload)
            if dependant.request_param_name:
                kwargs[dependant.request_param_name] = _Request(query)
        except ValidationError as error:
            return _json(422, {"detail": jsonable_encoder(error.errors(include_url=False))})

        try:
            result = route.endpoint(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except HTTPException as error:
            return _json(error.status_code, {"detail": error.detail})
        except ValidationError as error:
            return _json(422, {"detail": jsonable_encoder(error.errors(include_url=False))})
        except Exception as error:  # noqa: BLE001 - surfaced to the page, not swallowed
            return _json(500, {"detail": f"{type(error).__name__}: {error}"})

        if isinstance(result, Response):
            content_type = result.media_type or "application/octet-stream"
            return result.status_code, content_type, await _body_of(result)
        return _json(200, result)

    return _json(404, {"detail": f"no route for {method} {path}"})


async def dispatch_text(
    method: str, path: str, query_json: str, body_text: str | None
) -> tuple[int, str, bytes]:
    """:func:`dispatch` with string arguments, for callers across a language boundary.

    Passing JSON text rather than JavaScript objects keeps the conversion rules
    in one place - Python's ``json`` - instead of relying on how a foreign
    object proxy maps ``null``, nested arrays and integer-valued floats.
    """
    query = json.loads(query_json) if query_json else {}
    if isinstance(query, str):
        query = dict(parse_qsl(query))
    body = json.loads(body_text) if body_text else None
    return await dispatch(method, path, query, body)


def set_progress(callback: Callable[[str, int, int], None] | None) -> None:
    """Report long operations - rendering and analysis - to the caller."""
    from kerbside.api.session import get_session

    get_session().on_progress = callback
