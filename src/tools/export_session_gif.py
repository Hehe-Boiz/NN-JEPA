"""Export one recorded session as a preview GIF using only Pillow."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from data import settings


DEFAULT_FPS = 10
DEFAULT_EVERY_NTH_FRAME = 4
DEFAULT_MAX_FRAMES = 300
DEFAULT_WIDTH = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one session into a preview GIF.")
    parser.add_argument("--session-id", required=True, help="Example: session_20260605_150919")
    parser.add_argument(
        "--source",
        choices=["raw", "processed"],
        default="raw",
        help="Use raw camera frames or processed resized images.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--every-nth-frame", type=int, default=DEFAULT_EVERY_NTH_FRAME)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_frames_dir(session_id: str, source: str) -> Path:
    if source == "raw":
        return settings.RAW_DATA_DIR / session_id / settings.FRAME_DIR_NAME
    return settings.PROCESSED_IMAGE_DIR / session_id


def collect_frame_paths(frames_dir: Path, every_nth_frame: int, max_frames: int) -> list[Path]:
    if every_nth_frame < 1:
        raise ValueError("--every-nth-frame must be >= 1")
    if max_frames < 1:
        raise ValueError("--max-frames must be >= 1")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frames_dir}")

    frame_paths = [
        path
        for path in sorted(frames_dir.iterdir())
        if path.is_file() and path.suffix.lower() in settings.FRAME_EXTENSIONS
    ]
    sampled = frame_paths[::every_nth_frame]
    return sampled[:max_frames]


def resize_image(image: Image.Image, width: int) -> Image.Image:
    if width < 1:
        raise ValueError("--width must be >= 1")
    if image.width <= width:
        return image.copy()
    scale = width / image.width
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def export_gif(frame_paths: list[Path], output_path: Path, width: int, fps: int) -> None:
    if fps < 1:
        raise ValueError("--fps must be >= 1")
    if not frame_paths:
        raise RuntimeError("No frames found for GIF export")

    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(resize_image(image.convert("RGB"), width=width))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=False,
    )


def default_output_path(session_id: str) -> Path:
    return settings.DATA_ROOT / "previews" / f"{session_id}.gif"


def main() -> None:
    args = parse_args()
    frames_dir = resolve_frames_dir(args.session_id, args.source)
    frame_paths = collect_frame_paths(
        frames_dir=frames_dir,
        every_nth_frame=args.every_nth_frame,
        max_frames=args.max_frames,
    )
    output_path = args.output or default_output_path(args.session_id)
    export_gif(
        frame_paths=frame_paths,
        output_path=output_path,
        width=args.width,
        fps=args.fps,
    )
    print(output_path)


if __name__ == "__main__":
    main()
