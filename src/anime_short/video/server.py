from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from anime_short.config import AppConfig
from anime_short.schemas import CameraMove
from anime_short.video.animegen import AnimeGenGenerator

log = logging.getLogger(__name__)


class AnimeGenService:
    def __init__(self, config: AppConfig) -> None:
        self.generator = AnimeGenGenerator(config.video.animegen)
        self.lock = threading.Lock()
        self.loaded = False
        self.active_jobs = 0

    def load(self) -> None:
        if self.loaded:
            return
        self.generator.load()
        self.loaded = True

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.active_jobs += 1
            try:
                self.generator.width = int(payload["width"])
                self.generator.height = int(payload["height"])
                self.generator.fps = int(payload["fps"])
                result = self.generator.generate(
                    image=Path(payload["image"]),
                    prompt=str(payload.get("prompt") or ""),
                    camera=CameraMove(payload["camera"]),
                    duration=float(payload["duration"]),
                    output_path=Path(payload["output_path"]),
                )
                return {
                    "path": str(result.path),
                    "duration": result.duration,
                    "fps": result.fps,
                    "width": result.width,
                    "height": result.height,
                }
            finally:
                self.active_jobs -= 1

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "loaded": self.loaded,
            "busy": self.active_jobs > 0,
            "active_jobs": self.active_jobs,
        }


class AnimeGenHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], service: AnimeGenService) -> None:
        super().__init__(server_address, AnimeGenHandler)
        self.service = service


class AnimeGenHandler(BaseHTTPRequestHandler):
    server: AnimeGenHTTPServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json({"error": "not found"}, status=404)
            return
        self._send_json(self.server.service.health())

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("content-length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send_json(self.server.service.generate(payload))
        except Exception as exc:  # noqa: BLE001 - returned to the local client as JSON
            log.exception("animegen service request failed")
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("animegen service: " + fmt, *args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(config: AppConfig, host: str, port: int) -> None:
    service = AnimeGenService(config)
    log.info("animegen service: loading model")
    service.load()
    server = AnimeGenHTTPServer((host, port), service)
    log.info("animegen service: ready at http://%s:%d", host, port)
    server.serve_forever()
