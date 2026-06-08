"""Serve a simple local web app for browsing frame sessions as videos."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from data import settings
from models.vjepa21_presets import (
    DEFAULT_VJEPA21_FEATURE_PRESET,
    VJEPA21_FEATURE_PRESETS,
    vjepa21_feature_preset_options,
)


VIEWER_ROOT = Path(__file__).resolve().parents[1] / "viewer"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
MAX_JOB_LOG_LINES = 800
PROGRESS_PREFIX = "__JOB_PROGRESS__ "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local web viewer for recorded sessions.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def get_source_root(source: str) -> Path:
    if source == "raw":
        return settings.RAW_DATA_DIR
    if source == "processed":
        return settings.PROCESSED_IMAGE_DIR
    raise ValueError(f"Unsupported source: {source}")


def get_frames_dir(source: str, session_id: str) -> Path:
    if source == "raw":
        return settings.RAW_DATA_DIR / session_id / settings.FRAME_DIR_NAME
    if source == "processed":
        return settings.PROCESSED_IMAGE_DIR / session_id
    raise ValueError(f"Unsupported source: {source}")


def list_frame_paths(frames_dir: Path) -> list[Path]:
    if not frames_dir.exists():
        return []
    return [
        path
        for path in sorted(frames_dir.iterdir())
        if path.is_file() and path.suffix.lower() in settings.FRAME_EXTENSIONS
    ]


def collect_sessions(source: str) -> list[dict[str, Any]]:
    root = get_source_root(source)
    if not root.exists():
        return []

    sessions: list[dict[str, Any]] = []
    for session_dir in sorted(path for path in root.glob(settings.SESSION_GLOB) if path.is_dir()):
        frames_dir = get_frames_dir(source, session_dir.name)
        frame_paths = list_frame_paths(frames_dir)
        if not frame_paths:
            continue
        sessions.append(
            {
                "session_id": session_dir.name,
                "frame_count": len(frame_paths),
                "frame_size_hint": frame_paths[0].name,
                "files": describe_session_files(source=source, session_dir=session_dir),
            }
        )
    return sessions


def describe_session_files(source: str, session_dir: Path) -> list[str]:
    if source == "processed":
        return ["processed_images"]

    file_names = []
    for filename in (
        settings.SYNCED_ACTIONS_CSV_NAME,
        settings.SYNCED_IMU_CSV_NAME,
        settings.ACTIONS_CSV_NAME,
        settings.TELEMETRY_CSV_NAME,
        settings.ACCEL_CSV_NAME,
        settings.GYRO_CSV_NAME,
        settings.ROTVEC_CSV_NAME,
        settings.GPS_CSV_NAME,
    ):
        if (session_dir / filename).exists():
            file_names.append(filename)
    return file_names


def build_session_payload(source: str, session_id: str) -> dict[str, Any]:
    session_dir = get_source_root(source) / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    frame_paths = list_frame_paths(get_frames_dir(source, session_id))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found for session: {session_id}")

    return {
        "session_id": session_id,
        "source": source,
        "frame_count": len(frame_paths),
        "frames": [path.name for path in frame_paths],
        "files": describe_session_files(source=source, session_dir=session_dir),
    }


def read_static_file(name: str) -> bytes:
    path = VIEWER_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing viewer asset: {path}")
    return path.read_bytes()


def build_sync_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.sync_drive_data",
        "--check-zips",
    ]


def build_preprocess_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.preprocess_data",
    ]


def build_extract_feature_command(
    batch_size: int = 32,
    num_workers: int | None = None,
    encoder_preset: str = DEFAULT_VJEPA21_FEATURE_PRESET,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.extract_vjepa_features",
        "--vjepa-root",
        "vjepa2",
        "--encoder-preset",
        encoder_preset,
        "--manifest-dir",
        "data/processed/manifests",
        "--batch-size",
        str(batch_size),
        "--dtype",
        "fp32",
    ]
    if num_workers is not None:
        command += ["--num-workers", str(num_workers)]
    return command


def build_job_command(job_name: str, payload: dict[str, Any] | None = None) -> list[str]:
    payload = payload or {}
    if job_name == "sync":
        return build_sync_command()
    if job_name == "preprocess":
        return build_preprocess_command()
    if job_name == "extract_features":
        batch_size = int(payload.get("batch_size", 32))
        num_workers_value = payload.get("num_workers")
        encoder_preset = str(payload.get("feature_preset") or DEFAULT_VJEPA21_FEATURE_PRESET)
        num_workers = None if num_workers_value in (None, "") else int(num_workers_value)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if num_workers is not None and num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if encoder_preset not in VJEPA21_FEATURE_PRESETS:
            available = ", ".join(VJEPA21_FEATURE_PRESETS)
            raise ValueError(f"Unknown feature_preset={encoder_preset!r}. Available: {available}")
        return build_extract_feature_command(
            batch_size=batch_size,
            num_workers=num_workers,
            encoder_preset=encoder_preset,
        )
    raise ValueError(f"Unsupported job: {job_name}")


def initial_progress(job_name: str) -> dict[str, Any]:
    labels = {
        "sync": "Starting Drive sync",
        "preprocess": "Starting preprocessing",
        "extract_features": "Starting feature extraction",
    }
    return {
        "percent": 0.0,
        "label": labels.get(job_name, "Starting job"),
        "current": None,
        "total": None,
        "indeterminate": True,
    }


def completed_progress(status: str) -> dict[str, Any]:
    if status == "completed":
        return {
            "percent": 100.0,
            "label": "Completed",
            "current": None,
            "total": None,
            "indeterminate": False,
        }
    return {
        "percent": 0.0,
        "label": "Failed",
        "current": None,
        "total": None,
        "indeterminate": False,
    }


def parse_progress_line(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    percent = payload.get("percent")
    if percent is not None:
        payload["percent"] = max(0.0, min(float(percent), 100.0))
    return payload


class JobRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 1

    def start(self, job_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        command = build_job_command(job_name, payload)
        with self._lock:
            if self._job is not None and self._job["status"] == "running":
                raise RuntimeError(f"Job already running: {self._job['name']}")
            job_id = self._next_id
            self._next_id += 1
            self._job = {
                "id": job_id,
                "name": job_name,
                "status": "running",
                "command": command,
                "started_at": time.time(),
                "finished_at": None,
                "exit_code": None,
                "log": deque(maxlen=MAX_JOB_LOG_LINES),
                "progress": initial_progress(job_name),
            }
            self._append_log_locked(f"$ {' '.join(command)}")

        thread = threading.Thread(target=self._run, args=(job_id, command), daemon=True)
        thread.start()
        return self.snapshot()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if self._job is None or self._job["status"] != "running" or process is None:
                raise RuntimeError("No running job to cancel")
            self._append_log_locked("Cancel requested.")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                process.terminate()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                return {"job": None}
            job = dict(self._job)
            job["log"] = list(self._job["log"])
            return {"job": job}

    def _run(self, job_id: int, command: list[str]) -> None:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_ROOT) if not existing_pythonpath else str(SRC_ROOT) + os.pathsep + existing_pythonpath
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._process = process
            assert process.stdout is not None
            for line in process.stdout:
                clean_line = line.rstrip()
                progress = parse_progress_line(clean_line)
                with self._lock:
                    if self._job is not None and self._job["id"] == job_id:
                        if progress is not None:
                            self._job["progress"] = progress
                        else:
                            self._append_log_locked(clean_line)
            exit_code = process.wait()
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._job["exit_code"] = exit_code
                    self._job["finished_at"] = time.time()
                    self._job["status"] = "completed" if exit_code == 0 else "failed"
                    self._job["progress"] = completed_progress(self._job["status"])
                    self._append_log_locked(f"Job finished with exit code {exit_code}.")
                    self._process = None
        except Exception as exc:  # pragma: no cover - defensive guard for server runtime.
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._job["exit_code"] = None
                    self._job["finished_at"] = time.time()
                    self._job["status"] = "failed"
                    self._job["progress"] = completed_progress("failed")
                    self._append_log_locked(f"Job failed before process start: {exc}")
                    self._process = None

    def _append_log_locked(self, line: str) -> None:
        if self._job is not None:
            self._job["log"].append(line)


JOB_RUNNER = JobRunner()


class SessionViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(include_body=False)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch_post()

    def _dispatch(self, include_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._send_bytes(read_static_file("index.html"), "text/html; charset=utf-8", include_body)
                return
            if path == "/app.js":
                self._send_bytes(read_static_file("app.js"), "application/javascript; charset=utf-8", include_body)
                return
            if path == "/styles.css":
                self._send_bytes(read_static_file("styles.css"), "text/css; charset=utf-8", include_body)
                return
            if path == "/api/sessions":
                source = query_value(query, "source", default="raw")
                payload = {"sessions": collect_sessions(source)}
                self._send_json(payload, include_body)
                return
            if path == "/api/session":
                source = query_value(query, "source", default="raw")
                session_id = query_value(query, "session_id")
                payload = build_session_payload(source=source, session_id=session_id)
                self._send_json(payload, include_body)
                return
            if path == "/api/jobs":
                self._send_json(JOB_RUNNER.snapshot(), include_body)
                return
            if path == "/api/feature-presets":
                self._send_json({"presets": vjepa21_feature_preset_options()}, include_body)
                return
            if path.startswith("/media/"):
                self._serve_media(path, include_body)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _dispatch_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/jobs/start":
                payload = self._read_json_body()
                job_name = str(payload.get("job", ""))
                self._send_json(JOB_RUNNER.start(job_name, payload), include_body=True)
                return
            if path == "/api/jobs/cancel":
                self._send_json(JOB_RUNNER.cancel(), include_body=True)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except RuntimeError as exc:
            self.send_error(HTTPStatus.CONFLICT, str(exc))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _serve_media(self, path: str, include_body: bool) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) != 4:
            raise ValueError("Invalid media path")
        _, source, session_id, filename = parts
        frame_path = get_frames_dir(source, session_id) / filename
        if frame_path.suffix.lower() not in settings.FRAME_EXTENSIONS:
            raise ValueError(f"Unsupported media file: {filename}")
        resolved = frame_path.resolve()
        allowed_root = get_frames_dir(source, session_id).resolve()
        if resolved.parent != allowed_root or not resolved.exists():
            raise FileNotFoundError(f"Frame not found: {filename}")
        mime_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self._send_bytes(resolved.read_bytes(), mime_type, include_body)

    def _send_json(self, payload: dict[str, Any], include_body: bool) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", include_body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_bytes(self, data: bytes, content_type: str, include_body: bool) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)


def query_value(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    values = query.get(key)
    if not values:
        if default is not None:
            return default
        raise ValueError(f"Missing query parameter: {key}")
    return values[0]


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SessionViewerHandler)
    print(f"http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
