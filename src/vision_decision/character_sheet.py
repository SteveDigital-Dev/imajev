"""Deterministic character-sheet crops for the Jev-Vision Phase 1 proxy.

This intentionally handles the boxed-off first case: separated figures on a plain
or transparent background. It produces review candidates, not approved labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter

from .images import load_image

Box = tuple[int, int, int, int]


def _border_colour(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    step = max(1, 2 * (width + height) // 4096)
    pixels = []
    for x in range(0, width, step):
        pixels.extend((rgb.getpixel((x, 0)), rgb.getpixel((x, height - 1))))
    for y in range(0, height, step):
        pixels.extend((rgb.getpixel((0, y)), rgb.getpixel((width - 1, y))))
    return tuple(int(statistics.median(pixel[channel] for pixel in pixels)) for channel in range(3))


def _foreground_mask(image: Image.Image, tolerance: int) -> Image.Image:
    if not 0 <= tolerance <= 255:
        raise ValueError("background tolerance must be between 0 and 255")
    if "A" in image.getbands():
        alpha = image.getchannel("A")
        low, high = alpha.getextrema()
        if low < 250 and high > 8:
            return alpha.point(lambda value: 255 if value > 8 else 0, mode="1").convert("L")

    rgb = image.convert("RGB")
    background = _border_colour(rgb)
    mask = Image.new("L", rgb.size)
    pixels = rgb.get_flattened_data() if hasattr(rgb, "get_flattened_data") else rgb.getdata()
    mask.putdata([
        255 if max(abs(pixel[channel] - background[channel]) for channel in range(3)) > tolerance else 0
        for pixel in pixels
    ])
    return mask


def _components(mask: Image.Image, min_pixels: int) -> list[tuple[Box, int]]:
    width, height = mask.size
    data = bytearray(mask.tobytes())
    found: list[tuple[Box, int]] = []
    for start in range(width * height):
        if not data[start]:
            continue
        data[start] = 0
        queue = deque([start])
        x0 = x1 = start % width
        y0 = y1 = start // width
        count = 0
        while queue:
            index = queue.popleft()
            x, y = index % width, index // width
            count += 1
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
            if x and data[index - 1]:
                data[index - 1] = 0
                queue.append(index - 1)
            if x + 1 < width and data[index + 1]:
                data[index + 1] = 0
                queue.append(index + 1)
            if y and data[index - width]:
                data[index - width] = 0
                queue.append(index - width)
            if y + 1 < height and data[index + width]:
                data[index + width] = 0
                queue.append(index + width)
        if count >= min_pixels:
            found.append(((x0, y0, x1 + 1, y1 + 1), count))
    return found


def _pad(box: Box, ratio: float, size: tuple[int, int]) -> Box:
    if not 0 <= ratio <= 0.5:
        raise ValueError("padding ratio must be between 0 and 0.5")
    x0, y0, x1, y1 = box
    amount = math.ceil(max(x1 - x0, y1 - y0) * ratio)
    return max(0, x0 - amount), max(0, y0 - amount), min(size[0], x1 + amount), min(size[1], y1 + amount)


def _row_major(boxes: Iterable[Box]) -> list[Box]:
    boxes = list(boxes)
    if not boxes:
        return []
    median_height = statistics.median(y1 - y0 for _, y0, _, y1 in boxes)
    rows: list[dict[str, object]] = []
    for box in sorted(boxes, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        centre = (box[1] + box[3]) / 2
        for row in rows:
            if abs(centre - float(row["centre"])) <= median_height * 0.45:
                cast_boxes = row["boxes"]
                assert isinstance(cast_boxes, list)
                cast_boxes.append(box)
                row["centre"] = statistics.mean((item[1] + item[3]) / 2 for item in cast_boxes)
                break
        else:
            rows.append({"centre": centre, "boxes": [box]})
    ordered: list[Box] = []
    for row in sorted(rows, key=lambda item: float(item["centre"])):
        cast_boxes = row["boxes"]
        assert isinstance(cast_boxes, list)
        ordered.extend(sorted(cast_boxes, key=lambda item: item[0]))
    return ordered


def detect_character_boxes(
    image: Image.Image,
    *,
    background_tolerance: int = 28,
    min_area_ratio: float = 0.002,
    padding_ratio: float = 0.04,
    analysis_max_side: int = 1024,
    max_characters: int = 64,
) -> list[Box]:
    """Return padded character boxes in row-major order.

    Detection runs on a bounded analysis image, filters small disconnected marks,
    and maps boxes back to the original pixels. The caller must visually review the
    crops before using them as benchmark cases.
    """
    if not 0 < min_area_ratio < 1:
        raise ValueError("minimum area ratio must be between 0 and 1")
    if analysis_max_side < 64:
        raise ValueError("analysis max side must be at least 64")
    if max_characters < 1:
        raise ValueError("max characters must be positive")

    width, height = image.size
    scale = min(1.0, analysis_max_side / max(width, height))
    analysis_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    analysis = image if analysis_size == image.size else image.resize(analysis_size, Image.Resampling.LANCZOS)
    mask = _foreground_mask(analysis, background_tolerance)
    # Close one-pixel antialiasing gaps without bridging normal sheet gutters.
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    min_pixels = max(4, math.ceil(analysis.width * analysis.height * min_area_ratio))
    components = _components(mask, min_pixels)
    if not components:
        raise ValueError("no separated character candidates found; use a plain or transparent background")
    if len(components) > max_characters:
        raise ValueError(f"found {len(components)} candidates, above max_characters={max_characters}")

    mapped = []
    for (x0, y0, x1, y1), _ in components:
        original = (
            math.floor(x0 / scale),
            math.floor(y0 / scale),
            math.ceil(x1 / scale),
            math.ceil(y1 / scale),
        )
        mapped.append(_pad(original, padding_ratio, image.size))
    return _row_major(mapped)


def crop_character_sheet(source: Path, output: Path, **settings: object) -> dict[str, object]:
    """Create lossless crops and a hash-bound manifest in a new directory."""
    source, output = Path(source), Path(output)
    image, source_meta = load_image(source, preserve_alpha=True)
    boxes = detect_character_boxes(image, **settings)
    output.mkdir(parents=True, exist_ok=False)
    crops = []
    for index, box in enumerate(boxes, 1):
        filename = f"character-{index:03d}.png"
        path = output / filename
        image.crop(box).save(path, format="PNG", optimize=True)
        blob = path.read_bytes()
        crops.append(
            {
                "id": f"character-{index:03d}",
                "path": filename,
                "bbox_xyxy": list(box),
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "jev-vision-character-crops-v1",
        "source": {"name": source.name, **source_meta},
        "settings": {
            "background_tolerance": settings.get("background_tolerance", 28),
            "min_area_ratio": settings.get("min_area_ratio", 0.002),
            "padding_ratio": settings.get("padding_ratio", 0.04),
            "analysis_max_side": settings.get("analysis_max_side", 1024),
            "max_characters": settings.get("max_characters", 64),
        },
        "review_status": "unreviewed",
        "crops": crops,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crop separated characters from a character sheet")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background-tolerance", type=int, default=28)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--padding-ratio", type=float, default=0.04)
    parser.add_argument("--analysis-max-side", type=int, default=1024)
    parser.add_argument("--max-characters", type=int, default=64)
    args = parser.parse_args(argv)
    manifest = crop_character_sheet(
        args.source,
        args.output,
        background_tolerance=args.background_tolerance,
        min_area_ratio=args.min_area_ratio,
        padding_ratio=args.padding_ratio,
        analysis_max_side=args.analysis_max_side,
        max_characters=args.max_characters,
    )
    print(json.dumps({"output": str(args.output), "crops": len(manifest["crops"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
