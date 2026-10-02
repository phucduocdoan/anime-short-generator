"""ACE-Step 1.5 music generation over its local REST API (`/release_task`, `/query_result`,
`/v1/audio`). The server runs as its own process and keeps its model loaded, so nothing here
touches the GPU directly.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

STATUS_DONE = 1
STATUS_FAILED = 2


class AceStepClient:
    def __init__(self, options: dict[str, Any], client: httpx.Client | None = None) -> None:
        self.prompt = options["prompt"]
        self.bpm = options.get("bpm")
        self.key_scale = options.get("key_scale", "")
        self.poll_seconds = float(options.get("poll_seconds", 5))
        self.timeout_seconds = float(options.get("timeout_seconds", 900))
        self._client = client or httpx.Client(
            base_url=options.get("base_url", "http://127.0.0.1:8001"), timeout=60
        )

    def generate(self, duration: float, seed: int, output_path: Path) -> Path:
        payload = {
            "prompt": self.prompt,
            "lyrics": "[Instrumental]",
            "audio_duration": duration,
            "bpm": self.bpm,
            "key_scale": self.key_scale,
            "time_signature": "4",
            "thinking": False,
            "batch_size": 1,
            "use_random_seed": False,
            "seed": seed,
            "audio_format": "mp3",
        }
        task_id = self._post("/release_task", payload)["task_id"]
        log.info("acestep: task %s (%.0fs, seed %d)", task_id, duration, seed)
        file_url = self._wait(task_id)
        response = self._client.get(file_url)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path

    def _wait(self, task_id: str) -> str:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            item = self._post("/query_result", {"task_id_list": [task_id]})[0]
            if item["status"] == STATUS_DONE:
                return json.loads(item["result"])[0]["file"]
            if item["status"] == STATUS_FAILED:
                raise RuntimeError(f"acestep task {task_id} failed: {item.get('result')}")
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"acestep task {task_id} not done after {self.timeout_seconds:.0f}s")

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()["data"]

    def close(self) -> None:
        self._client.close()
