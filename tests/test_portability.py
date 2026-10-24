import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from core import DEFAULT_CONFIG, validate_config, validate_review
from segment import load_config

import install

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/sam3-segmentation"


def test_installed_skill_is_self_contained(tmp_path):
    destination = install.install(tmp_path / "installed-skills" / "sam3-segmentation")
    assert (destination / "SKILL.md").exists()
    assert (destination / "LICENSE").exists()
    assert (destination / "uv.lock").exists()
    assert (destination / "environments/sam3/uv.lock").exists()
    assert (destination / "scripts/run").exists()
    assert (destination / "agents/openai.yaml").exists()
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
    destination = tmp_path / "not-created" / "sam3-segmentation"
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


def skill_frontmatter():
    text = (SKILL / "SKILL.md").read_text()
    assert text.startswith("---\n")
    # A strict YAML parser, as Pi and the reference validator use: an unquoted
    # ": " inside the description once made Pi skip the skill silently.
    return yaml.safe_load(text.split("\n---\n", 1)[0][4:])


def test_skill_metadata_follows_agent_skills_spec():
    fields = skill_frontmatter()
    allowed = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    assert set(fields) <= allowed
    assert fields["name"] == SKILL.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"]) and len(fields["name"]) <= 64
    assert 1 <= len(fields["description"]) <= 1024
    assert 1 <= len(fields["compatibility"]) <= 500
    assert all(isinstance(v, str) for v in fields["metadata"].values())
    assert len((SKILL / "SKILL.md").read_text().splitlines()) < 500


def test_skill_redistribution_license():
    assert skill_frontmatter()["license"] == "MIT"
    assert (SKILL / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_versions_and_names_agree_everywhere():
    version = skill_frontmatter()["metadata"]["version"]
    plugin = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    marketplace = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert plugin["version"] == version
    assert plugin["name"] == SKILL.name
    assert [entry["name"] for entry in marketplace["plugins"]] == [SKILL.name]
    for project in [SKILL, SKILL / "environments/sam3"]:
        assert (
            tomllib.loads((project / "pyproject.toml").read_text())["project"]["version"] == version
        )
    assert f"## {version}" in (ROOT / "CHANGELOG.md").read_text()
    openai = yaml.safe_load((SKILL / "agents/openai.yaml").read_text())
    assert len(openai["interface"]["short_description"]) <= 64


LAUNCHER = SKILL / "scripts/run"


def run_launcher(*args, home):
    return subprocess.run(
        ["sh", str(LAUNCHER), *args],
        env={**os.environ, "SAM3_SEGMENTATION_HOME": str(home)},
        capture_output=True,
        text=True,
    )


def test_launcher_keeps_environments_outside_the_skill(tmp_path):
    base = run_launcher("base", "path", home=tmp_path).stdout.strip()
    model = run_launcher("model", "path", home=tmp_path).stdout.strip()
    assert Path(base).parent == Path(model).parent == tmp_path / "envs"
    assert Path(base).name.startswith("base-") and Path(model).name.startswith("model-")
    assert SKILL not in Path(base).parents


def test_launcher_environment_name_follows_the_lock(tmp_path):
    copy = tmp_path / "skill"
    shutil.copytree(SKILL, copy, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    launcher = copy / "scripts/run"
    env = {**os.environ, "SAM3_SEGMENTATION_HOME": str(tmp_path / "home")}
    before = subprocess.run(
        ["sh", launcher, "base", "path"], env=env, capture_output=True, text=True
    )
    with (copy / "uv.lock").open("a") as lock:
        lock.write("\n")
    after = subprocess.run(
        ["sh", launcher, "base", "path"], env=env, capture_output=True, text=True
    )
    assert before.stdout != after.stdout


def test_launcher_rejects_bad_arguments(tmp_path):
    assert run_launcher("gpu", "sam.py", home=tmp_path).returncode == 2
    assert run_launcher("base", home=tmp_path).returncode == 2
    missing = run_launcher("base", "no-such-script.py", home=tmp_path)
    assert missing.returncode == 2 and "no such script" in missing.stderr


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_launcher_runs_base_tools_without_writing_into_the_skill(tmp_path):
    before = sorted(p.relative_to(SKILL) for p in SKILL.rglob("*") if ".venv" not in p.parts)
    result = run_launcher("base", "segment.py", "doctor", home=tmp_path)
    assert result.returncode == 0, result.stderr
    assert '"torch_available": false' in result.stdout
    after = sorted(p.relative_to(SKILL) for p in SKILL.rglob("*") if ".venv" not in p.parts)
    assert before == after
    assert list((tmp_path / "envs").glob("base-*"))


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
