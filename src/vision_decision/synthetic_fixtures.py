"""Deterministic synthetic Phase 1 protocol fixtures for Jev-Vision.

Generates clearly labeled character-sheet proxy assets with versioned closed rubrics,
source-family split isolation, and hash-bound manifests. All outputs are marked as
pipeline fixtures, not model evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont

from .images import load_image
from .character_sheet import detect_character_boxes, crop_character_sheet


RUBRIC_VERSION = "jev-vision-phase1-rubric-v1"

USABLE_RUBRIC = {
    "id": "usable",
    "type": "boolean",
    "question": "Is this crop usable for evaluation (intact, unclipped, unoccluded, in focus, complete views)?",
    "yes_description": "Crop shows one complete character view without clipping, occlusion, blur, or missing major parts.",
    "no_description": "Crop is clipped at sheet edge, has occlusion, blur, missing views, or does not contain one complete character.",
}

QUALITY_RUBRIC = {
    "id": "quality",
    "type": "ordinal",
    "question": "Rate the visual quality of this generated character crop.",
    "levels": [
        {"value": 1, "description": "Unusable: severe artifacts, incoherent geometry, unrecognizable as character."},
        {"value": 2, "description": "Poor: major artifacts, distorted proportions, broken anatomy, heavy noise."},
        {"value": 3, "description": "Fair: visible artifacts or inconsistencies, but character is recognizable and mostly coherent."},
        {"value": 4, "description": "Good: minor artifacts only, consistent proportions, clean geometry, faithful to prompt."},
        {"value": 5, "description": "Excellent: artifact-free, anatomically correct, consistent style, high visual fidelity."},
    ],
}

STYLE_MATCH_RUBRIC = {
    "id": "style_match",
    "type": "ordinal",
    "question": "How well does this crop match the target style reference?",
    "levels": [
        {"value": 1, "description": "No match: style is completely different from reference."},
        {"value": 2, "description": "Weak: only vague resemblance; major style elements missing or wrong."},
        {"value": 3, "description": "Partial: some style elements present but inconsistent or mixed with other styles."},
        {"value": 4, "description": "Strong: clear style adherence with minor deviations."},
        {"value": 5, "description": "Exact: indistinguishable from reference style; all elements faithful."},
    ],
}

CATEGORY_VOCABULARY = [
    "humanoid",
    "creature",
    "mech",
    "vehicle",
    "prop",
    "environment",
    "effect",
    "unknown",
]

CATEGORY_RUBRIC = {
    "id": "category",
    "type": "choice",
    "question": "What category of generated asset does this crop depict?",
    "options": [{"value": v, "description": v.capitalize()} for v in CATEGORY_VOCABULARY if v != "unknown"]
    + [{"value": "unknown", "description": "Does not fit any defined category or cannot be determined."}],
}

ALL_RUBRICS = {
    "usable": USABLE_RUBRIC,
    "quality": QUALITY_RUBRIC,
    "style_match": STYLE_MATCH_RUBRIC,
    "category": CATEGORY_RUBRIC,
}


@dataclass(frozen=True)
class SyntheticFamily:
    """A source family for split isolation."""
    name: str
    style_seed: int
    character_count: int
    canvas_size: tuple[int, int]
    background_color: tuple[int, int, int, int]
    character_template: str


@dataclass
class SyntheticAsset:
    """A generated synthetic character sheet asset."""
    family: SyntheticFamily
    index: int
    image: Image.Image
    crops: list[Image.Image]
    crop_boxes: list[tuple[int, int, int, int]]
    crop_hashes: list[str]
    source_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def asset_id(self) -> str:
        return f"{self.family.name}-{self.index:03d}"

    @property
    def source_cluster(self) -> str:
        return self.family.name


@dataclass
class FixtureRecord:
    """A benchmark record for a synthetic fixture."""
    id: str
    group_id: str
    track: Literal["visual"]
    family: str
    split: Literal["dev", "calibration", "test"]
    images: list[dict[str, str]]
    request: dict[str, Any]
    gold: Any
    annotation_status: Literal["draft"] = "draft"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "track": self.track,
            "family": self.family,
            "split": self.split,
            "images": self.images,
            "request": self.request,
            "gold": self.gold,
            "annotation_status": self.annotation_status,
            "provenance": self.provenance,
        }


def _deterministic_color(seed: int, index: int) -> tuple[int, int, int, int]:
    """Generate a deterministic color from seed and index."""
    h = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b, 255)


def _draw_character(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], template: str, color: tuple[int, int, int, int]) -> None:
    """Draw a simple character representation in the given box."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    w, h = x1 - x0, y1 - y0
    r = min(w, h) // 3

    if template == "humanoid":
        # Head
        draw.ellipse((cx - r, cy - h // 3 - r, cx + r, cy - h // 3 + r), fill=color)
        # Torso
        draw.rectangle((cx - r, cy - h // 3 + r, cx + r, cy + h // 3), fill=color)
        # Arms
        draw.rectangle((cx - r - w // 4, cy - h // 6, cx - r, cy + h // 6), fill=color)
        draw.rectangle((cx + r, cy - h // 6, cx + r + w // 4, cy + h // 6), fill=color)
        # Legs
        draw.rectangle((cx - r // 2, cy + h // 3, cx - r // 2 + r, cy + h // 2), fill=color)
        draw.rectangle((cx + r // 2, cy + h // 3, cx + r // 2 + r, cy + h // 2), fill=color)
    elif template == "creature":
        # Quadruped body
        draw.ellipse((cx - w // 3, cy - h // 4, cx + w // 3, cy + h // 4), fill=color)
        # Head
        draw.ellipse((cx + w // 3 - r // 2, cy - h // 4 - r, cx + w // 3 + r // 2, cy - h // 4), fill=color)
        # Legs
        for lx in (cx - w // 4, cx + w // 4):
            draw.rectangle((lx - r // 2, cy + h // 4, lx + r // 2, cy + h // 2), fill=color)
    elif template == "mech":
        # Angular torso
        draw.polygon([
            (cx - w // 3, cy - h // 3),
            (cx + w // 3, cy - h // 3),
            (cx + w // 2, cy + h // 3),
            (cx - w // 2, cy + h // 3),
        ], fill=color)
        # Head sensor
        draw.rectangle((cx - r // 2, cy - h // 3 - r, cx + r // 2, cy - h // 3), fill=(255, 255, 0, 255))
        # Limbs
        draw.rectangle((cx - w // 3 - r, cy - h // 6, cx - w // 3, cy + h // 6), fill=color)
        draw.rectangle((cx + w // 3, cy - h // 6, cx + w // 3 + r, cy + h // 6), fill=color)
    else:
        # Default: simple rectangle
        draw.rectangle(box, fill=color)


def generate_synthetic_sheet(family: SyntheticFamily, asset_index: int) -> SyntheticAsset:
    """Generate a deterministic synthetic character sheet."""
    canvas_w, canvas_h = family.canvas_size
    image = Image.new("RGBA", (canvas_w, canvas_h), family.background_color)
    draw = ImageDraw.Draw(image)

    chars_per_row = max(1, int(math.sqrt(family.character_count)))
    cell_w = canvas_w // chars_per_row
    cell_h = canvas_h // math.ceil(family.character_count / chars_per_row)

    crop_boxes = []
    crops = []

    for i in range(family.character_count):
        row = i // chars_per_row
        col = i % chars_per_row
        x0 = col * cell_w + cell_w // 8
        y0 = row * cell_h + cell_h // 8
        x1 = (col + 1) * cell_w - cell_w // 8
        y1 = (row + 1) * cell_h - cell_h // 8

        color = _deterministic_color(family.style_seed, asset_index * family.character_count + i)
        _draw_character(draw, (x0, y0, x1, y1), family.character_template, color)

        box = (x0, y0, x1, y1)
        crop_boxes.append(box)
        crops.append(image.crop(box))

    source_bytes = image.tobytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    crop_hashes = [hashlib.sha256(c.tobytes()).hexdigest() for c in crops]

    return SyntheticAsset(
        family=family,
        index=asset_index,
        image=image,
        crops=crops,
        crop_boxes=crop_boxes,
        crop_hashes=crop_hashes,
        source_hash=source_hash,
        metadata={
            "generator": "jev-vision-synthetic-fixture",
            "rubric_version": RUBRIC_VERSION,
            "family": family.name,
            "asset_index": asset_index,
        },
    )


def _split_for_asset(asset: SyntheticAsset, fractions: dict[str, float], salt: str) -> Literal["dev", "calibration", "test"]:
    """Deterministically assign an asset to a split by its source cluster."""
    anchor = asset.source_cluster
    point = int(hashlib.sha256(f"{salt}\0{anchor}".encode()).hexdigest()[:12], 16) / 16**12
    edge = 0.0
    for split in ("dev", "calibration", "test"):
        edge += fractions.get(split, 0.0)
        if point < edge:
            return split
    return "test"


def _family_split_assignments(
    assets: list[SyntheticAsset], fractions: dict[str, float], salt: str
) -> dict[str, Literal["dev", "calibration", "test"]]:
    """Allocate whole source families deterministically, with split coverage when possible."""
    splits = ("dev", "calibration", "test")
    if set(fractions) != set(splits) or any(not isinstance(fractions[key], (int, float)) or fractions[key] < 0 for key in splits):
        raise ValueError("split_fractions must contain non-negative dev, calibration, and test values")
    total = math.fsum(fractions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("split_fractions must sum to 1.0")
    families = sorted(
        {asset.source_cluster for asset in assets},
        key=lambda name: hashlib.sha256(f"{salt}\0{name}".encode()).hexdigest(),
    )
    count = len(families)
    raw = {key: fractions[key] * count for key in splits}
    quotas = {key: math.floor(raw[key]) for key in splits}
    for key in sorted(splits, key=lambda name: (-(raw[name] - quotas[name]), name))[: count - sum(quotas.values())]:
        quotas[key] += 1
    positive = [key for key in splits if fractions[key] > 0]
    if count >= len(positive):
        for missing in (key for key in positive if quotas[key] == 0):
            donor = max((key for key in splits if quotas[key] > 1), key=lambda key: (quotas[key], fractions[key]))
            quotas[donor] -= 1
            quotas[missing] += 1
    result: dict[str, Literal["dev", "calibration", "test"]] = {}
    index = 0
    for split in splits:
        for family in families[index:index + quotas[split]]:
            result[family] = split
        index += quotas[split]
    return result


def _make_request(field_id: str, rubric: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a request dict for a single decision field."""
    return {
        "schema_version": "1.0",
        "request_id": field_id,
        "state": state or {},
        "fields": [rubric],
        "execution": {"mode": "inspect", "allow_external_fallback": False},
    }


def create_fixture_records(
    assets: list[SyntheticAsset],
    split_fractions: dict[str, float],
    split_salt: str,
) -> list[FixtureRecord]:
    """Create benchmark records from synthetic assets with split isolation."""
    records = []
    family_splits = _family_split_assignments(assets, split_fractions, split_salt)

    for asset in assets:
        split = family_splits[asset.source_cluster]

        for crop_idx, (crop, box, crop_hash) in enumerate(zip(asset.crops, asset.crop_boxes, asset.crop_hashes)):
            crop_id = f"{asset.asset_id}-crop-{crop_idx:03d}"
            group_id = f"grp-{asset.source_cluster}-{asset.index:03d}-{crop_idx:03d}"

            # Deterministic synthetic labels (construction_verified route)
            # These are pipeline fixtures, not human-reviewed model evidence
            usable_gold = True
            quality_gold = 4
            style_match_gold = 4
            category_gold = asset.family.character_template

            provenance = {
                "source_clusters": [asset.source_cluster],
                "image_sources": [{
                    "source_id": asset.asset_id,
                    "synthetic": True,
                    "generator": "jev-vision-synthetic-fixture",
                    "rubric_version": RUBRIC_VERSION,
                }],
                "synthetic": True,
                "fixture_type": "pipeline_fixture",
                "not_model_evidence": True,
                "label_origin": "construction_verified: deterministic synthetic fixture",
                "review_protocol": "imajev-bench-blind-review-v1",
                "construction": {
                    "truth_source": "generator_spec",
                    "truth": {
                        "usable": usable_gold,
                        "quality": quality_gold,
                        "style_match": style_match_gold,
                        "category": category_gold,
                    },
                    "image_checks": "passed",
                },
                "review_route": "construction_verified",
            }

            # Create one record per rubric field per crop
            for field_name, rubric in ALL_RUBRICS.items():
                field_id = f"{crop_id}-{field_name}"
                if field_name == "usable":
                    gold = usable_gold
                elif field_name == "quality":
                    gold = quality_gold
                elif field_name == "style_match":
                    gold = style_match_gold
                elif field_name == "category":
                    gold = category_gold
                else:
                    gold = None

                records.append(FixtureRecord(
                    id=field_id,
                    group_id=group_id,
                    track="visual",
                    family=asset.family.name,
                    split=split,
                    images=[{"path": f"assets/{asset.asset_id}/crop-{crop_idx:03d}.png", "sha256": crop_hash}],
                    request=_make_request(field_id, rubric, {"rubric_version": RUBRIC_VERSION, "crop_box": box}),
                    gold=gold,
                    provenance=provenance.copy(),
                ))

    return records


def write_fixture_dataset(
    records: list[FixtureRecord],
    assets: list[SyntheticAsset],
    output: Path,
    split_fractions: dict[str, float] | None = None,
    split_salt: str = "jev-vision-synthetic-phase1-salt-v1",
) -> dict[str, Any]:
    """Write fixture dataset with assets and records."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory {output} already exists; refusing to overwrite")

    output.mkdir(parents=True)
    assets_dir = output / "assets"
    assets_dir.mkdir()

    # Write asset images and compute hashes from saved files
    asset_manifest = {}
    crop_hashes_by_asset: dict[str, list[str]] = {}
    for asset in assets:
        asset_dir = assets_dir / asset.asset_id
        asset_dir.mkdir()
        source_path = asset_dir / "sheet.png"
        asset.image.save(source_path, format="PNG", optimize=True)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

        crop_hashes = []
        for crop_idx, crop in enumerate(asset.crops):
            crop_path = asset_dir / f"crop-{crop_idx:03d}.png"
            crop.save(crop_path, format="PNG", optimize=True)
            crop_hash = hashlib.sha256(crop_path.read_bytes()).hexdigest()
            crop_hashes.append(crop_hash)

        crop_hashes_by_asset[asset.asset_id] = crop_hashes
        asset_manifest[asset.asset_id] = {
            "source_hash": source_hash,
            "crop_hashes": crop_hashes,
            "crop_boxes": asset.crop_boxes,
            "metadata": asset.metadata,
        }

    # Update records with correct hashes from saved files
    updated_records = []
    for record in records:
        record_dict = record.to_dict()
        # Extract asset_id from image path
        image_path = record_dict["images"][0]["path"]
        # path format: assets/asset-id/crop-XXX.png
        asset_id = image_path.split("/")[1]
        crop_idx = int(image_path.split("-")[-1].split(".")[0])
        record_dict["images"][0]["sha256"] = crop_hashes_by_asset[asset_id][crop_idx]
        updated_records.append(record_dict)

    # Write records.jsonl
    records_path = output / "records.jsonl"
    with records_path.open("w") as f:
        for record_dict in updated_records:
            f.write(json.dumps(record_dict, allow_nan=False) + "\n")

    # Write assembly manifest
    split_counts = defaultdict(int)
    for r in updated_records:
        split_counts[r["split"]] += 1

    if split_fractions is None:
        split_fractions = {"dev": 0.2, "calibration": 0.1, "test": 0.7}
    assembly = {
        "dataset": "jev-vision-synthetic-phase1-fixtures",
        "version": RUBRIC_VERSION,
        "records": len(updated_records),
        "unique_assets": len(assets),
        "unique_families": len({a.family.name for a in assets}),
        "split_records": dict(split_counts),
        "split_fractions": split_fractions,
        "split_salt": split_salt,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "fixture_type": "pipeline_fixture",
        "not_model_evidence": True,
        "rubric_version": RUBRIC_VERSION,
    }
    (output / "assembly.json").write_text(json.dumps(assembly, indent=2) + "\n")

    # Write asset manifest
    (output / "asset_manifest.json").write_text(json.dumps(asset_manifest, indent=2) + "\n")

    return assembly


def get_default_families() -> list[SyntheticFamily]:
    """Get the default set of synthetic families for Phase 1 fixtures."""
    return [
        SyntheticFamily(
            name="fantasy-humanoid",
            style_seed=0xF1A7A5F1,
            character_count=4,
            canvas_size=(512, 512),
            background_color=(240, 240, 240, 255),
            character_template="humanoid",
        ),
        SyntheticFamily(
            name="scifi-mech",
            style_seed=0x5C1F1F12,
            character_count=3,
            canvas_size=(512, 512),
            background_color=(20, 20, 30, 255),
            character_template="mech",
        ),
        SyntheticFamily(
            name="creature-bestiary",
            style_seed=0xB3A57B33,
            character_count=5,
            canvas_size=(512, 512),
            background_color=(245, 245, 240, 255),
            character_template="creature",
        ),
        SyntheticFamily(
            name="modern-human",
            style_seed=0xA0D3E5F4,
            character_count=4,
            canvas_size=(512, 512),
            background_color=(250, 250, 250, 255),
            character_template="humanoid",
        ),
    ]


def build_phase1_fixtures(
    output: Path,
    *,
    families: list[SyntheticFamily] | None = None,
    assets_per_family: int = 3,
    split_fractions: dict[str, float] | None = None,
    split_salt: str = "jev-vision-synthetic-phase1-salt-v1",
) -> dict[str, Any]:
    """Build the complete Phase 1 synthetic fixture dataset."""
    if families is None:
        families = get_default_families()

    if split_fractions is None:
        split_fractions = {"dev": 0.2, "calibration": 0.1, "test": 0.7}

    # Generate all assets
    all_assets = []
    for family in families:
        for i in range(assets_per_family):
            all_assets.append(generate_synthetic_sheet(family, i))

    # Create records with split isolation
    records = create_fixture_records(all_assets, split_fractions, split_salt)

    # Write dataset
    assembly = write_fixture_dataset(records, all_assets, output, split_fractions, split_salt)

    return assembly


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic synthetic Phase 1 protocol fixtures for Jev-Vision"
    )
    parser.add_argument("output", type=Path, nargs="?", help="Output directory for fixture dataset (required unless --list-rubrics)")
    parser.add_argument("--families", type=int, default=4, help="Number of default families to use (max 4)")
    parser.add_argument("--assets-per-family", type=int, default=3, help="Assets to generate per family")
    parser.add_argument("--split-salt", type=str, default="jev-vision-synthetic-phase1-salt-v1")
    parser.add_argument("--list-rubrics", action="store_true", help="Print rubric definitions and exit")
    args = parser.parse_args(argv)

    if args.list_rubrics:
        print(json.dumps(ALL_RUBRICS, indent=2))
        return 0

    if args.output is None:
        parser.error("output is required unless --list-rubrics is used")

    families = get_default_families()[:args.families]
    split_fractions = {"dev": 0.2, "calibration": 0.1, "test": 0.7}

    assembly = build_phase1_fixtures(
        args.output,
        families=families,
        assets_per_family=args.assets_per_family,
        split_fractions=split_fractions,
        split_salt=args.split_salt,
    )

    print(json.dumps({"output": str(args.output), **assembly}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
