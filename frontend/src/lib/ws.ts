import { useEffect, useRef } from "react";
import type { Telemetry } from "./api";

/**
 * Subscribe to per-frame telemetry.
 *
 * Reconnects with a backoff rather than giving up: the backend restarts during
 * development, and a console that needs a manual reload to come back is a
 * console nobody trusts. The socket carries only small JSON - frames come over
 * MJPEG, because pushing 2.7 MB per frame down this socket would starve exactly
 * the messages the interface needs to stay responsive.
 */
export function useTelemetry(onMessage: (t: Telemetry) => void): void {
  const handler = useRef(onMessage);
  handler.current = onMessage;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;
    let delay = 500;

    const connect = (): void => {
      if (closed) return;
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${location.host}/ws/telemetry`);
      socket.onopen = () => {
        delay = 500;
      };
      socket.onmessage = (event) => {
        try {
          handler.current(JSON.parse(event.data as string) as Telemetry);
        } catch {
          /* a malformed frame is not worth tearing the socket down for */
        }
      };
      socket.onclose = () => {
        if (closed) return;
        timer = window.setTimeout(connect, delay);
        delay = Math.min(delay * 2, 8000);
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      closed = true;
      if (timer) window.clearTimeout(timer);
      socket?.close();
    };
  }, []);
}
