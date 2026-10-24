# Repository guidance

Read `skills/sam3-segmentation/SKILL.md` and its references before changing the workflow.
The skill folder must remain self-contained when copied out of this repository.

- CPU/base tools: `uv run --locked --project skills/sam3-segmentation --group dev pytest tests`
- Lint: `uv run --locked --project skills/sam3-segmentation --group dev ruff check skills/sam3-segmentation/scripts tests install.py`
- Validate: `uvx --from "git+https://github.com/agentskills/agentskills#subdirectory=skills-ref" skills-ref validate skills/sam3-segmentation` and `claude plugin validate --strict .`
- Commands in the skill's documents go through `sh "$S/scripts/run" base|model ...`, which keeps environments outside the skill folder. Do not document bare `uv run --project "$S"` calls for users.
- GPU calls must be explicit and use the `model` environment (`skills/sam3-segmentation/environments/sam3`).
- Ordinary tests must not download weights, require credentials, contact providers or change accepted annotations.
- Preserve raw candidates before gates, native coordinates, pixel-tight boxes, and proposal-only status.
- No photographs, weights, credentials, home-directory paths or private run outputs in commits. Eval fixtures must be public domain or CC0 and listed in `evals/fixtures/CREDITS.md`.
- Demo fixtures are synthetic and must never be described as model-quality evidence.
- Changing code/runtime identity invalidates inference caches; do not remove that safeguard.
- Keep model terms separate from this repository's MIT licence.
- The data-format identifiers (`detailed-segmentation/1`, `-review/1`, `-decisions/1`) predate the skill's name; keep them so existing runs stay readable.

## Releasing

1. Bump the version in `SKILL.md` (`metadata.version`), `.claude-plugin/plugin.json` and both `pyproject.toml` files, and add a `CHANGELOG.md` section. A test checks they agree. Claude Code users only receive an update when the plugin version changes.
2. If `SKILL.md`'s description changed, rerun the trigger evals on Haiku, Sonnet and Opus (`evals/README.md`). If its instructions changed, rerun the behaviour evals.
3. Tag `vX.Y.Z` and publish a GitHub release with the changelog section.
