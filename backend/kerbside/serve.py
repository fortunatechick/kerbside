"""``python -m kerbside.serve`` - the whole application in one process.

FastAPI serves the REST API, the MJPEG stream, the telemetry WebSocket and the
built frontend from the same origin. One origin means no CORS configuration to
get wrong and no second server to start; one process means no Node on the target
machine.

If the frontend has not been built, the root route says so and explains how -
rather than returning a 404 that looks like a broken install.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from kerbside.api.routes import router

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

_NO_BUILD = """<!doctype html><html><head><meta charset="utf-8">
<title>Kerbside &mdash; frontend not built</title>
<style>body{background:#0B0F14;color:#E8EEF4;font:15px/1.6 system-ui,sans-serif;
padding:60px 24px;max-width:640px;margin:0 auto}
code{background:#1A2230;border:1px solid #2A3644;border-radius:4px;padding:2px 6px;
font-family:ui-monospace,Menlo,monospace}
h1{font-size:26px;margin:0 0 12px}p{color:#98A6B5}</style></head><body>
<h1>The API is running; the console is not built</h1>
<p>The backend is serving on this port. To build the interface:</p>
<p><code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></p>
<p>Then reload. The REST API is available now at <code>/api/status</code>, and the
interactive documentation at <a href="/docs" style="color:#3FA7D6">/docs</a>.</p>
</body></html>"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kerbside",
        version="0.1.0",
        summary="Traffic analytics from a fixed roadside camera, classical DIP only.",
        description=(
            "Vehicle counts, speeds, lane assignment, size class and colour from "
            "pixels and a one-time geometric calibration. This system measures "
            "speed; it is not a speed enforcement device and makes no claim to "
            "evidentiary standing."
        ),
    )
    app.include_router(router)

    if DIST.is_dir():
        assets = DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(DIST / "index.html")

        # response_model=None: FastAPI otherwise tries to build a Pydantic
        # response model from the return annotation, and a union of Response
        # subclasses is not a valid field type. This only bites once dist/
        # exists, which is why it is worth a comment rather than a silent fix.
        @app.get("/{path:path}", include_in_schema=False, response_model=None)
        def spa(path: str) -> FileResponse | HTMLResponse:
            # Client-side routing: anything that is not a real file falls back
            # to the app shell, so a deep link still loads.
            candidate = DIST / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(DIST / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def missing() -> HTMLResponse:
            return HTMLResponse(_NO_BUILD)

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kerbside.serve", description="Run the Kerbside console."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--open", action="store_true", help="open a browser once serving")
    parser.add_argument(
        "--scenario",
        default=None,
        help="load a synthetic scenario at start-up, so the console opens on data",
    )
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args(argv)

    if args.scenario:
        from kerbside.api.session import get_session

        print(f"rendering scenario {args.scenario} ({args.duration:g} s)...")
        get_session().load_scenario(args.scenario, duration_s=args.duration)

    url = f"http://{args.host}:{args.port}/"
    print(f"Kerbside serving on {url}")
    if args.open:
        webbrowser.open(url)

    import uvicorn

    uvicorn.run(
        "kerbside.serve:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
