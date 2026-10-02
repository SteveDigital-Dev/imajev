import hashlib
import json

import pytest
from PIL import Image, ImageDraw

from vision_decision.character_sheet import crop_character_sheet, detect_character_boxes


def test_detects_transparent_characters_in_row_major_order():
    image = Image.new("RGBA", (240, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 69, 89), fill=(220, 30, 30, 255))
    draw.rectangle((140, 15, 199, 94), fill=(30, 220, 30, 255))
    draw.rectangle((25, 115, 84, 164), fill=(30, 30, 220, 255))

    boxes = detect_character_boxes(image, min_area_ratio=0.01, padding_ratio=0)

    assert boxes == [(20, 20, 70, 90), (140, 15, 200, 95), (25, 115, 85, 165)]


def test_opaque_background_filters_small_noise():
    image = Image.new("RGB", (220, 100), (248, 248, 246))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 10, 74, 89), fill=(40, 50, 60))
    draw.rectangle((135, 15, 204, 84), fill=(90, 70, 140))
    draw.point((110, 50), fill=(0, 0, 0))

    boxes = detect_character_boxes(
        image, background_tolerance=20, min_area_ratio=0.01, padding_ratio=0
    )

    assert boxes == [(15, 10, 75, 90), (135, 15, 205, 85)]


def test_crop_manifest_binds_source_and_outputs_and_refuses_overwrite(tmp_path):
    source = tmp_path / "sheet.png"
    image = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    # Black foreground proves the file intake preserves alpha; flattening this
    # image to RGB would make the character indistinguishable from its background.
    ImageDraw.Draw(image).rectangle((20, 10, 89, 69), fill=(0, 0, 0, 255))
    image.save(source)
    output = tmp_path / "crops"

    manifest = crop_character_sheet(source, output, min_area_ratio=0.01, padding_ratio=0)

    saved = json.loads((output / "manifest.json").read_text())
    crop = output / "character-001.png"
    assert saved == manifest
    assert saved["review_status"] == "unreviewed"
    assert saved["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert saved["crops"][0]["sha256"] == hashlib.sha256(crop.read_bytes()).hexdigest()
    assert saved["crops"][0]["bbox_xyxy"] == [20, 10, 90, 70]
    with pytest.raises(FileExistsError):
        crop_character_sheet(source, output, min_area_ratio=0.01, padding_ratio=0)


def test_blank_sheet_is_rejected():
    with pytest.raises(ValueError, match="no separated character candidates"):
        detect_character_boxes(Image.new("RGB", (100, 100), "white"))
