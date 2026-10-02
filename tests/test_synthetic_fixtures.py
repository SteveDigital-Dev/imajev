"""Tests for the deterministic synthetic Phase 1 fixture builder."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from vision_decision.synthetic_fixtures import (
    RUBRIC_VERSION,
    ALL_RUBRICS,
    USABLE_RUBRIC,
    QUALITY_RUBRIC,
    STYLE_MATCH_RUBRIC,
    CATEGORY_RUBRIC,
    CATEGORY_VOCABULARY,
    SyntheticFamily,
    generate_synthetic_sheet,
    create_fixture_records,
    write_fixture_dataset,
    build_phase1_fixtures,
    get_default_families,
    _split_for_asset,
    _make_request,
    FixtureRecord,
)


def test_rubric_version_is_frozen():
    assert RUBRIC_VERSION == "jev-vision-phase1-rubric-v1"


def test_all_four_rubrics_present():
    assert set(ALL_RUBRICS.keys()) == {"usable", "quality", "style_match", "category"}


def test_usable_rubric_is_boolean_with_descriptions():
    assert USABLE_RUBRIC["type"] == "boolean"
    assert "yes_description" in USABLE_RUBRIC
    assert "no_description" in USABLE_RUBRIC
    assert USABLE_RUBRIC["id"] == "usable"


def test_quality_rubric_is_ordinal_five_levels():
    assert QUALITY_RUBRIC["type"] == "ordinal"
    assert len(QUALITY_RUBRIC["levels"]) == 5
    values = [l["value"] for l in QUALITY_RUBRIC["levels"]]
    assert values == [1, 2, 3, 4, 5]
    assert all("description" in l for l in QUALITY_RUBRIC["levels"])
    assert QUALITY_RUBRIC["id"] == "quality"


def test_style_match_rubric_is_ordinal_five_levels():
    assert STYLE_MATCH_RUBRIC["type"] == "ordinal"
    assert len(STYLE_MATCH_RUBRIC["levels"]) == 5
    values = [l["value"] for l in STYLE_MATCH_RUBRIC["levels"]]
    assert values == [1, 2, 3, 4, 5]
    assert STYLE_MATCH_RUBRIC["id"] == "style_match"


def test_category_rubric_is_choice_with_closed_vocab_plus_unknown():
    assert CATEGORY_RUBRIC["type"] == "choice"
    option_values = {o["value"] for o in CATEGORY_RUBRIC["options"]}
    assert option_values == set(CATEGORY_VOCABULARY)
    assert "unknown" in option_values
    assert CATEGORY_RUBRIC["id"] == "category"


def test_category_vocabulary_is_frozen():
    expected = ["humanoid", "creature", "mech", "vehicle", "prop", "environment", "effect", "unknown"]
    assert CATEGORY_VOCABULARY == expected


def test_deterministic_generation_same_seed_same_output():
    family = SyntheticFamily(
        name="test-family",
        style_seed=42,
        character_count=3,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset1 = generate_synthetic_sheet(family, 0)
    asset2 = generate_synthetic_sheet(family, 0)
    assert asset1.source_hash == asset2.source_hash
    assert asset1.crop_hashes == asset2.crop_hashes


def test_different_assets_different_hashes():
    family = SyntheticFamily(
        name="test-family",
        style_seed=42,
        character_count=2,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset1 = generate_synthetic_sheet(family, 0)
    asset2 = generate_synthetic_sheet(family, 1)
    assert asset1.source_hash != asset2.source_hash


def test_split_assignment_is_deterministic_by_source_cluster():
    family = SyntheticFamily(
        name="split-test",
        style_seed=1,
        character_count=2,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset = generate_synthetic_sheet(family, 0)
    fractions = {"dev": 0.2, "calibration": 0.1, "test": 0.7}
    salt = "test-salt"

    split1 = _split_for_asset(asset, fractions, salt)
    split2 = _split_for_asset(asset, fractions, salt)
    assert split1 == split2
    assert split1 in ("dev", "calibration", "test")


def test_split_isolation_same_family_same_split():
    family = SyntheticFamily(
        name="iso-family",
        style_seed=123,
        character_count=2,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    assets = [generate_synthetic_sheet(family, i) for i in range(5)]
    fractions = {"dev": 0.2, "calibration": 0.1, "test": 0.7}
    salt = "iso-salt"

    splits = [_split_for_asset(a, fractions, salt) for a in assets]
    # All assets from same family (same source_cluster) must get same split
    assert len(set(splits)) == 1


def test_different_families_can_get_different_splits():
    families = [
        SyntheticFamily(name=f"fam-{i}", style_seed=i, character_count=2,
                        canvas_size=(256, 256), background_color=(255, 255, 255, 255),
                        character_template="humanoid")
        for i in range(10)
    ]
    assets = [generate_synthetic_sheet(f, 0) for f in families]
    fractions = {"dev": 0.2, "calibration": 0.1, "test": 0.7}
    salt = "multi-fam-salt"

    splits = [_split_for_asset(a, fractions, salt) for a in assets]
    # With 10 families, we should see some distribution across splits
    assert len(set(splits)) >= 2


def test_fixture_records_have_correct_structure(tmp_path):
    family = SyntheticFamily(
        name="record-test",
        style_seed=999,
        character_count=2,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset = generate_synthetic_sheet(family, 0)
    records = create_fixture_records([asset], {"dev": 0.2, "calibration": 0.1, "test": 0.7}, "test-salt")

    assert len(records) == 2 * 4  # 2 crops * 4 rubrics

    for record in records:
        assert isinstance(record, FixtureRecord)
        assert record.track == "visual"
        assert record.family == "record-test"
        assert record.split in ("dev", "calibration", "test")
        assert len(record.images) == 1
        assert "path" in record.images[0]
        assert "sha256" in record.images[0]
        assert record.request["schema_version"] == "1.0"
        assert len(record.request["fields"]) == 1
        assert record.annotation_status == "draft"
        assert record.provenance["synthetic"] is True
        assert record.provenance["label_origin"] == "construction_verified: deterministic synthetic fixture"
        assert record.provenance["review_route"] == "construction_verified"
        assert "construction" in record.provenance
        assert record.provenance["construction"]["truth_source"] == "generator_spec"


def test_fixture_records_gold_matches_rubric_domains():
    family = SyntheticFamily(
        name="gold-test",
        style_seed=777,
        character_count=1,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="creature",
    )
    asset = generate_synthetic_sheet(family, 0)
    records = create_fixture_records([asset], {"dev": 1.0, "calibration": 0.0, "test": 0.0}, "salt")

    for record in records:
        field = record.request["fields"][0]
        gold = record.gold
        if field["type"] == "boolean":
            assert isinstance(gold, bool)
        elif field["type"] == "ordinal":
            assert isinstance(gold, int)
            assert gold in [l["value"] for l in field["levels"]]
        elif field["type"] == "choice":
            assert isinstance(gold, str)
            assert gold in [o["value"] for o in field["options"]]


def test_write_fixture_dataset_creates_expected_files(tmp_path):
    family = SyntheticFamily(
        name="write-test",
        style_seed=111,
        character_count=2,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset = generate_synthetic_sheet(family, 0)
    records = create_fixture_records([asset], {"dev": 1.0, "calibration": 0.0, "test": 0.0}, "salt")
    output = tmp_path / "fixture-out"

    assembly = write_fixture_dataset(records, [asset], output)

    assert output.exists()
    assert (output / "records.jsonl").exists()
    assert (output / "assembly.json").exists()
    assert (output / "asset_manifest.json").exists()
    assert (output / "assets" / "write-test-000" / "sheet.png").exists()
    assert (output / "assets" / "write-test-000" / "crop-000.png").exists()
    assert (output / "assets" / "write-test-000" / "crop-001.png").exists()

    assert assembly["fixture_type"] == "pipeline_fixture"
    assert assembly["not_model_evidence"] is True
    assert assembly["rubric_version"] == RUBRIC_VERSION
    assert assembly["records"] == len(records)


def test_write_fixture_dataset_refuses_overwrite(tmp_path):
    family = SyntheticFamily(
        name="overwrite-test",
        style_seed=222,
        character_count=1,
        canvas_size=(128, 128),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset = generate_synthetic_sheet(family, 0)
    records = create_fixture_records([asset], {"dev": 1.0, "calibration": 0.0, "test": 0.0}, "salt")
    output = tmp_path / "fixture-overwrite"

    write_fixture_dataset(records, [asset], output)

    with pytest.raises(FileExistsError):
        write_fixture_dataset(records, [asset], output)


def test_asset_manifest_binds_hashes(tmp_path):
    family = SyntheticFamily(
        name="manifest-test",
        style_seed=333,
        character_count=2,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="mech",
    )
    asset = generate_synthetic_sheet(family, 0)
    records = create_fixture_records([asset], {"dev": 1.0, "calibration": 0.0, "test": 0.0}, "salt")
    output = tmp_path / "fixture-manifest"

    write_fixture_dataset(records, [asset], output)

    manifest = json.loads((output / "asset_manifest.json").read_text())
    assert asset.asset_id in manifest
    # Hashes are computed from saved PNG files, verify they match the actual files
    source_path = output / "assets" / asset.asset_id / "sheet.png"
    assert manifest[asset.asset_id]["source_hash"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    for crop_idx, crop_hash in enumerate(manifest[asset.asset_id]["crop_hashes"]):
        crop_path = output / "assets" / asset.asset_id / f"crop-{crop_idx:03d}.png"
        assert crop_hash == hashlib.sha256(crop_path.read_bytes()).hexdigest()
    # Crop boxes are serialized as lists in JSON
    assert manifest[asset.asset_id]["crop_boxes"] == [list(b) for b in asset.crop_boxes]


def test_records_jsonl_validates_against_schema(tmp_path):
    from imajev_bench.schema import validate_records

    family = SyntheticFamily(
        name="schema-test",
        style_seed=444,
        character_count=2,
        canvas_size=(256, 256),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset = generate_synthetic_sheet(family, 0)
    records = create_fixture_records([asset], {"dev": 1.0, "calibration": 0.0, "test": 0.0}, "salt")
    output = tmp_path / "fixture-schema"

    write_fixture_dataset(records, [asset], output)

    record_dicts = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    validated = validate_records(record_dicts, output, require_reviewed=False)
    assert len(validated) == len(records)


def test_build_phase1_fixtures_end_to_end(tmp_path):
    output = tmp_path / "phase1-fixtures"
    assembly = build_phase1_fixtures(
        output,
        families=get_default_families()[:2],
        assets_per_family=2,
        split_fractions={"dev": 0.2, "calibration": 0.1, "test": 0.7},
        split_salt="e2e-test-salt",
    )

    assert output.exists()
    assert assembly["records"] > 0
    assert assembly["unique_assets"] == 4  # 2 families * 2 assets
    assert assembly["unique_families"] == 2
    assert assembly["fixture_type"] == "pipeline_fixture"
    assert assembly["not_model_evidence"] is True
    assert assembly["split_salt"] == "e2e-test-salt"

    # Verify split isolation: all crops from same asset in same split
    record_dicts = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    asset_splits = {}
    for r in record_dicts:
        asset_id = r["id"].split("-crop-")[0]
        if asset_id in asset_splits:
            assert asset_splits[asset_id] == r["split"], f"Asset {asset_id} leaks across splits"
        else:
            asset_splits[asset_id] = r["split"]


def test_get_default_families_returns_four_families():
    families = get_default_families()
    assert len(families) == 4
    names = {f.name for f in families}
    assert names == {"fantasy-humanoid", "scifi-mech", "creature-bestiary", "modern-human"}


def test_default_family_allocation_covers_all_splits(tmp_path):
    assembly = build_phase1_fixtures(
        tmp_path / "all-splits",
        families=get_default_families(),
        assets_per_family=1,
        split_fractions={"dev": 0.2, "calibration": 0.1, "test": 0.7},
        split_salt="coverage-test-salt",
    )
    assert set(assembly["split_records"]) == {"dev", "calibration", "test"}


def test_make_request_builds_valid_request():
    request = _make_request("test-id", USABLE_RUBRIC, {"custom": "state"})
    assert request["schema_version"] == "1.0"
    assert request["request_id"] == "test-id"
    assert request["state"] == {"custom": "state"}
    assert len(request["fields"]) == 1
    assert request["fields"][0] == USABLE_RUBRIC
    assert request["execution"]["mode"] == "inspect"


def test_cli_list_rubrics(capsys):
    from vision_decision.synthetic_fixtures import main
    main(["--list-rubrics"])
    captured = capsys.readouterr()
    rubrics = json.loads(captured.out)
    assert set(rubrics.keys()) == {"usable", "quality", "style_match", "category"}


def test_cli_builds_fixtures(tmp_path):
    from vision_decision.synthetic_fixtures import main
    output = tmp_path / "cli-fixtures"
    main([str(output), "--families", "2", "--assets-per-family", "1"])
    assert output.exists()
    assert (output / "records.jsonl").exists()
    assert (output / "assembly.json").exists()


def test_fixture_provenance_marks_not_model_evidence():
    family = SyntheticFamily(
        name="provenance-test",
        style_seed=555,
        character_count=1,
        canvas_size=(128, 128),
        background_color=(255, 255, 255, 255),
        character_template="humanoid",
    )
    asset = generate_synthetic_sheet(family, 0)
    records = create_fixture_records([asset], {"dev": 1.0, "calibration": 0.0, "test": 0.0}, "salt")

    for record in records:
        assert record.provenance["synthetic"] is True
        assert record.provenance["label_origin"] == "construction_verified: deterministic synthetic fixture"
        assert record.provenance["review_route"] == "construction_verified"
        assert record.provenance["construction"]["truth_source"] == "generator_spec"
