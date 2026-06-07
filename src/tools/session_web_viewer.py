"""Serve a simple local web app for browsing frame sessions as videos."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from data import settings


VIEWER_ROOT = Path(__file__).resolve().parents[1] / "viewer"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


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


class SessionViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(include_body=False)

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
            if path.startswith("/media/"):
                self._serve_media(path, include_body)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
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
