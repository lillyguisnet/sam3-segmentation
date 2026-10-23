import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from core import DEFAULT_CONFIG, validate_config, validate_review
from segment import load_config

import install

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/detailed-segmentation"


def test_installed_skill_is_self_contained(tmp_path):
    destination = install.install(tmp_path / "installed-skills" / "detailed-segmentation")
    assert (destination / "SKILL.md").exists()
    assert (destination / "LICENSE").exists()
    assert (destination / "uv.lock").exists()
    assert (destination / "environments/sam3/uv.lock").exists()
    assert not list(destination.rglob(".venv"))
    output = tmp_path / "outside-repository/demo"
    result = subprocess.run(
        [sys.executable, str(destination / "scripts/segment.py"), "demo", "--out", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Exported 77" in result.stdout
    assert (output / "export/index.html").exists()
    with pytest.raises(FileExistsError):
        install.install(destination)


def test_installer_dry_run_does_not_create_directory(tmp_path):
    destination = tmp_path / "not-created" / "detailed-segmentation"
    install.install(destination, dry_run=True)
    assert not destination.exists()


def test_installer_preserves_standard_skill_name(tmp_path):
    with pytest.raises(ValueError, match="must be named"):
        install.install(tmp_path / "wrong-name")


def test_all_document_relative_links_resolve():
    for path in [ROOT / "README.md", ROOT / "TESTING.md", *SKILL.rglob("*.md")]:
        if ".venv" in path.parts:
            continue
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            assert target.exists(), f"Broken link in {path}: {link}"


def test_skill_metadata_and_redistribution_license():
    frontmatter = (SKILL / "SKILL.md").read_text().split("---", 2)[1]
    fields = dict(line.split(": ", 1) for line in frontmatter.strip().splitlines())
    assert fields["name"] == SKILL.name
    assert 1 <= len(fields["description"]) <= 1024
    assert fields["license"] == "MIT"
    assert (SKILL / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_no_private_machine_paths_or_credentials_in_sources():
    for path in [
        ROOT / "README.md",
        ROOT / "install.py",
        *SKILL.rglob("*.py"),
        *SKILL.rglob("*.md"),
    ]:
        if ".venv" in path.parts:
            continue
        text = path.read_text()
        assert not re.search(r"/(?:home|Users)/[A-Za-z0-9_.-]+", text)
        assert "family-ai/" not in text
        assert "127.0.0.1:9300" not in text
        assert "hf_" + "token=" not in text


@pytest.mark.parametrize(
    "changes",
    [
        {"passes": [{"name": "bad", "size": 12.5, "overlap": 2, "grid_step": None}]},
        {"passes": [{"name": "bad", "size": 12, "overlap": 2, "grid_step": True}]},
        {"passes": [None]},
        {"whole_objects": [{"label": "ear", "phrase": 42}]},
        {"seed": True},
        {"seed": -1},
    ],
)
def test_config_rejects_invalid_structures(changes):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(changes)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("selection", "point_quality_min", float("nan")),
        ("selection", "text_score_min", 2),
        ("selection", "point_stability_min", True),
        ("selection", "max_area", 12.3),
        ("refinement", "area_ratio_min", -1),
        ("refinement", "crop_size", 1),
        ("refinement", "stability_min", float("inf")),
        ("budget", "max_jobs", 1.5),
        ("budget", "max_grid_points", True),
    ],
)
def test_config_rejects_invalid_values(section, key, value):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("name", ["dog", "jars", "insects", "mushrooms", "cows"])
def test_non_corn_profiles_override_task_specific_defaults(name):
    path = SKILL / "configs" / "non-corn" / f"{name}.json"
    raw = json.loads(path.read_text())
    config = load_config(path)
    assert {"label", "phrases", "whole_objects", "passes", "selection", "refinement"} <= raw.keys()
    assert {"min_area", "max_area"} <= raw["selection"].keys()
    assert "crop_size" in raw["refinement"]
    assert "corn" not in json.dumps(config).lower()
    assert config["label"] != "kernel"
    assert config["selection"]["max_area"] != 650
    assert config["refinement"]["crop_size"] > 120


def test_example_configs_are_valid():
    for name in ["corn-starting-point.json", "smoke.json", "shapes-smoke.json"]:
        load_config(SKILL / "configs" / name)


@pytest.fixture
def review():
    return {
        "schema": "detailed-segmentation-review/1",
        "image_sha256": "abc",
        "coordinate_space": "native",
        "reviewer": "test",
        "observations": [
            {
                "type": "missing",
                "points": [[1, 2]],
                "negative_points": [],
                "reason": "Visible target",
                "uncertainty": "medium",
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "no-reviewer",
        "no-reason",
        "null-points",
        "unknown-field",
        "boolean-point",
        "no-target",
        "bad-container",
    ],
)
def test_strict_review_schema(review, mutation):
    if mutation == "no-reviewer":
        del review["reviewer"]
    elif mutation == "no-reason":
        del review["observations"][0]["reason"]
    elif mutation == "null-points":
        review["observations"][0]["points"] = None
    elif mutation == "unknown-field":
        review["observations"][0]["execute"] = "never"
    elif mutation == "boolean-point":
        review["observations"][0]["points"] = [[True, 1]]
    elif mutation == "no-target":
        review["observations"][0]["points"] = []
    else:
        review["observations"] = "not a list"
    with pytest.raises(ValueError):
        validate_review(review, {"canonical_sha256": "abc", "width": 480, "height": 500})
