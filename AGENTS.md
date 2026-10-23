# Repository guidance

Read `skills/detailed-segmentation/SKILL.md` and its references before changing the workflow.
The skill folder must remain self-contained when copied out of this repository.

- CPU/base tools: `uv run --locked --project skills/detailed-segmentation --group dev pytest tests`
- Lint: `uv run --locked --project skills/detailed-segmentation --group dev ruff check skills/detailed-segmentation/scripts tests install.py`
- GPU calls must be explicit and use `skills/detailed-segmentation/environments/sam3`.
- Ordinary tests must not download weights, require credentials, contact providers or change accepted annotations.
- Preserve raw candidates before gates, native coordinates, pixel-tight boxes, and proposal-only status.
- No photographs, weights, credentials, home-directory paths or private run outputs in commits.
- Demo fixtures are synthetic and must never be described as model-quality evidence.
- Changing code/runtime identity invalidates inference caches; do not remove that safeguard.
- Keep model terms separate from this repository's MIT licence.
