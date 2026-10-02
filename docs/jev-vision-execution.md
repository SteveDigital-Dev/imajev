# Jev-Vision execution status

Updated: 2026-10-02

Pi routing was smoke-tested through Ollama Cloud on this date:
`nemotron-3-ultra:cloud` is the bounded evaluator and
`gpt-oss:120b-cloud` is its fallback. Neither route is part of the visual model
benchmark itself.

Active reviewer: **OpenCode (Nemotron 3 Ultra)**. Claude is on cooldown.

## Reconciled status — 2026-10-02

- The focused intake, fixture, contract, benchmark-schema, scoring, and
  calibration suite passes: **101 tests**.
- An installed `jev-synthetic-fixtures` smoke run produced 64 records across
  dev, calibration, and test with source-family isolation and a hash-bound
  receipt. These are deterministic pipeline fixtures, not model evidence.
- The full legacy repository suite was attempted but does not currently
  collect in the lightweight Python 3.11 environment: optional ML/web/data
  dependencies are absent, and `scripts/p2/p2_common.py` contains a pre-existing
  Python 3.11 f-string syntax error. This does not invalidate the 101 focused
  tests, but it prevents claiming a repository-wide green suite.

## Contract

The project asks configurable yes/no, choice, and ordinal quality questions about generated visual assets and returns a probability for every allowed answer plus `unknown`. The first target is generated 3D character and video quality.

The existing repository already supplies the typed request contract, one- or two-image inference, post-hoc calibration, Brier/ECE scoring, family macro accuracy, abstention metrics, hash-bound benchmark records, paired evidence clusters, and latency reporting. It does not yet provide character-sheet intake, a reviewed 3D evaluation set, video frame pooling, or measured candidate comparison runs.

## Milestones

| Milestone | Status | Evidence |
| --- | --- | --- |
| M0: inspect existing imajev capability | complete | Typed contracts and benchmark scoring are under `src/vision_decision/` and `src/imajev_bench/`. |
| M1: deterministic character-sheet intake | implemented, locally tested | `jev-sheet-crop` writes individual PNG crops and a source-hash-bound `manifest.json`; tests cover transparent and opaque sheets, noise filtering, ordering, and overwrite refusal. |
| M1a: deterministic synthetic Phase 1 fixtures | implemented, tested | `jev-synthetic-fixtures` builds deterministic proxy assets with closed rubrics (usable, quality, style_match, category + unknown), source-family split isolation, hash-bound manifests, and pipeline-fixture marking. CLI and focused tests in `tests/test_synthetic_fixtures.py`. |
| M2: reviewed Phase 1 proxy set | blocked on real assets and labels | The synthetic fixture rubric and split machinery are frozen; representative character sheets and independent human review are still required. |
| M3: calibrated Phase 1 model result | blocked on M2 and model runtime | Fit only on the calibration split; publish ECE, Brier, family macro accuracy, abstention, and edge latency with a run receipt. |
| M4: paired multi-view candidate trial | blocked on held-out 3D cases and candidate endpoints | Use identical cases across Laya Vision, Kev, and the selected third candidate; freeze frame selection and pooling before the test run. |

## M1 usage

Install the development environment with Python 3.11 or newer, then run:

```sh
jev-sheet-crop path/to/sheet.png --output out/character-crops
```

The command expects a plain or transparent background and separated characters. It creates a new output directory, one lossless PNG per detected character, and `manifest.json`. The manifest records the source digest, source dimensions, detection settings, crop boxes, and crop digests. Existing output is refused so a prior intake cannot be silently replaced.

Review every crop before admitting it to the proxy set. Detection is deterministic preprocessing; it does not establish that a crop contains one complete character.

## M1a usage

Generate deterministic synthetic Phase 1 fixtures (pipeline fixtures, not model evidence):

```sh
jev-synthetic-fixtures out/phase1-fixtures --families 4 --assets-per-family 3
```

The command creates a dataset directory with:
- `assets/` — synthetic character sheets and crops (PNG, lossless)
- `records.jsonl` — benchmark records with draft gold labels from construction spec
- `assembly.json` — dataset receipt with split counts, hashes, and fixture metadata
- `asset_manifest.json` — per-asset source and crop hashes

All outputs are marked `"fixture_type": "pipeline_fixture"` and `"not_model_evidence": true`. Rubrics are versioned (`jev-vision-phase1-rubric-v1`) and closed: `usable` (boolean), `quality` (ordinal 1–5), `style_match` (ordinal 1–5), `category` (choice from frozen vocabulary + `unknown`). Splits are isolated by source family (dev 20%, calibration 10%, test 70%).

List rubric definitions:

```sh
jev-synthetic-fixtures --list-rubrics
```

## Phase 1 acceptance protocol

Each admitted crop receives four separately scored records:

1. `usable`: boolean, with a written rejection rubric for clipping, occlusion, blur, and missing views.
2. `quality`: five-level ordinal score with anchored descriptions.
3. `style_match`: five-level ordinal score against a named target style or fixed reference. The target must be part of the model input.
4. `category`: choice from a frozen closed vocabulary plus `unknown`.

Split by the original generated asset, not by crop, so related views cannot cross dev, calibration, and test. Subjective fields require independent review under the repository's annotation protocol. Keep calibration and test assets untouched after the split is frozen.

## Phase 2 comparison protocol

- Use the exact same held-out assets, frame order, resize policy, prompts, and typed answer domains for every candidate.
- Record model/checkpoint identity, execution interface, hardware, warmup, concurrency, per-case latency, failures, and raw probability vectors.
- Predeclare a pooling rule for multi-view assets. Report per-frame and pooled results separately.
- Report distribution-argmax ECE, Brier score, family macro accuracy, coverage/abstention, and p50/p95 edge latency. Use paired evidence-cluster analysis for candidate differences.
- Keep absolute rubric scores distinct from preferences among the candidates in the trial.

## True blockers to project completion

- No representative, licensed set of generated character sheets and multi-view 3D assets is present in the repository.
- The synthetic fixture rubric is frozen, but its construction labels are not
  evidence for real assets; representative usability, quality, style, and
  category labels still require independent review.
- Laya Vision and Kev endpoints/checkpoints are not configured here; the third comparison candidate is not selected.
- The serving path accepts at most two images and explicitly rejects animated or multi-frame files, so a video sampling and pooling interface still needs implementation and evaluation.
- End-to-end calibration and latency claims require real model runs on declared edge hardware. Unit and schema tests cannot supply that evidence.
