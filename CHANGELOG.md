# Changelog

Versions follow [Semantic Versioning](https://semver.org/). Before 1.0, a minor
version (0.2, 0.3, ...) may change commands or file formats; the changelog says
how to move over.

## 0.1.0 — 2026-10-24

First public release.

- Agent skill `sam3-segmentation` in the open [Agent Skills](https://agentskills.io)
  format, installable with `npx skills`, as a Claude Code plugin, or by copying
  the folder (`install.py`).
- Direct SAM 3 calls: every instance of a phrase or example box, one object from
  clicks or a box, automatic "segment everything", and batches with one model load.
- Full workflow for detailed annotation: overlapping crops, raw candidates saved
  before any filter, score audits, review sheets and flags, recorded reversible
  review decisions, guarded boundary refinement, pixel-checked PNG / COCO RLE
  exports and a zoomable viewer.
- A demo on generated shapes that needs no GPU, model or account.
- Locked environments: a small base environment and a pinned SAM 3 / PyTorch
  2.10 (CUDA 12.8) environment, created by `scripts/run` outside the skill folder.
- Agent evals for `claude plugin eval` (`evals/`): when the skill should and
  should not be used, and four safety behaviours, each compared with no skill.
  They led to two rules: state the download size (about 11 GB) when asking to
  install, and never use or repeat a token the user pastes.
- Renamed from the unpublished `detailed-segmentation`. Saved run files keep
  their `detailed-segmentation/...` format names and stay readable.

Known limitations, planned for later versions:

- The skill has no rule for when a review is finished (for example "stop when a
  round finds nothing new") and no time or round budget.
- It does not tell the agent which reviewer to use; an agent may silently use its
  own default model as the "independent" reviewer.
- Nothing stops an agent from removing many proposals at once without having
  inspected each one.
- A single boundary refinement cannot be undone on its own; it can only be marked
  uncertain.
- Model inference needs Linux (or WSL2) and an NVIDIA GPU. CPU and Apple GPU
  inference are not implemented.
