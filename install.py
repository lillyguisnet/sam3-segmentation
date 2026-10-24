#!/usr/bin/env python3
"""Copy the complete skill into an agent's skill directory; never overwrite it."""

import argparse
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "skills" / "sam3-segmentation"
TARGETS = {
    "codex": Path("~/.agents/skills/sam3-segmentation"),
    "claude": Path("~/.claude/skills/sam3-segmentation"),
    "pi": Path("~/.pi/agent/skills/sam3-segmentation"),
}


def install(destination, dry_run=False):
    destination = Path(destination).expanduser().absolute()
    if destination.name != SOURCE.name:
        raise ValueError(
            "The skill directory must be named sam3-segmentation to match its frontmatter"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing skill: {destination}")
    if SOURCE == destination or SOURCE in destination.parents:
        raise ValueError("Destination must be outside the source skill")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            SOURCE,
            destination,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "*.pyc",
            ),
        )
    print(("Would copy" if dry_run else "Installed") + f" complete skill to {destination}")
    print(
        "No models downloaded, credentials read, or agent settings changed. Restart/reload the agent if needed."
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", choices=TARGETS)
    group.add_argument("--dest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        install(args.dest or TARGETS[args.target], args.dry_run)
    except (FileExistsError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
