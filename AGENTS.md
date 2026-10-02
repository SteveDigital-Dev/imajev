# Jev-Vision agent team

## Objective

Build a reproducible evaluator for generated 3D assets and video. Start with single-character crops from character sheets, then compare the same held-out multi-view cases across candidate vision systems. Every scored answer must use the existing typed decision contract and report calibrated probabilities.

## Team

- **Codex — implementation lead.** Own code, tests, dataset tooling, run receipts, and integration. Keep changes small enough to audit and never claim a model result from schema or unit tests alone.
- **OpenCode (`opencode/nemotron-3-ultra-free`) — active reviewer and research critic.** Review protocols, labels, failure cases, and claims. Challenge leakage, candidate-relative scores, weak rubrics, and calibration fitted on test data. Record findings before a release decision.
- **Claude — reviewer (on cooldown).** Not currently active in this session. Previously served as reviewer and research critic.
- **Pi — bounded evaluator.** Use Ollama Cloud model `nemotron-3-ultra:cloud`; fall back to `gpt-oss:120b-cloud`. Run fixed prompts against frozen artifacts, return structured review notes, and do not change labels or source files during an evaluation pass.

Aider is excluded from this project. Gemini is deprecated for this workflow.

## Working agreement

1. Preserve raw inputs. Derived crops, labels, predictions, and summaries go to new paths and carry source hashes.
2. Split by source asset or asset family before tuning. Every candidate sees identical held-out cases in identical order.
3. Define absolute rubrics before collecting labels. Never convert a ranking among current candidates into an absolute probability of quality.
4. Fit calibration on the calibration split only. Report ECE, Brier score, family macro accuracy, abstention behavior, and measured edge latency on the untouched test split.
5. Treat single-frame and pooled multi-frame results as separate interfaces. Record the frame selection and pooling rule in each run receipt.
6. Codex runs the relevant tests; OpenCode reviews the protocol and claims while Claude is on cooldown; Pi reviews a bounded, frozen packet. Human approval remains required for subjective quality and style labels.

## Current execution file

Read `docs/jev-vision-execution.md` before starting work. Update its evidence and blockers when a milestone changes status.
