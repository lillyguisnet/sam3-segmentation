import copy
import io
from argparse import Namespace
from zipfile import ZipFile

import numpy as np
import pytest
import segment
from core import decode, digest, encode, file_sha, read_json, write_json
from PIL import Image


class FakeBackend:
    """Geometry fixture only; never evidence of real-model segmentation quality."""

    identity = {"backend": "synthetic-test", "version": 1}

    def __init__(self):
        self.calls = 0
        self.fail_points = False

    def text(self, image, phrase):
        self.calls += 1
        masks = []
        for number, score in enumerate([0.9, 0.8, 0.01]):
            mask = np.zeros((image.height, image.width), bool)
            if number == 0:
                mask[3:11, 4:12] = True
            elif number == 1:
                mask[-1, -1] = True
            else:
                mask[30:35, 40:45] = True
            masks.append(
                {"rle": encode(mask), "score": score, "stability": None, "hypothesis": number}
            )
        return masks

    def points(self, image, prompts):
        self.calls += 1
        if self.fail_points:
            raise RuntimeError("simulated interruption")
        result = []
        for i, prompt in enumerate(prompts):
            x, y = map(int, prompt["points"][0])
            mask = np.zeros((image.height, image.width), bool)
            mask[
                max(0, y - 1) : min(image.height, y + 2), max(0, x - 1) : min(image.width, x + 2)
            ] = True
            result.append(
                {
                    "rle": encode(mask),
                    "score": 0.99,
                    "stability": 0.95,
                    "prompt_index": i,
                    "hypothesis": 0,
                }
            )
        return result


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (80, 60), (80, 90, 100)).save(source)
    config = segment.load_config()
    config["passes"] = [{"name": "test", "size": None, "overlap": 0, "grid_step": 20}]
    config["phrases"] = ["synthetic object"]
    config["whole_objects"] = []
    config["selection"].update(min_area=1, max_area=None)
    run = tmp_path / "run"
    segment.prepare(source, run, config)
    backend = FakeBackend()
    monkeypatch.setattr(segment, "backend_from", lambda args, manifest: backend)
    args = Namespace(review=None, confirm_review_points=False, limit=None)
    return run, backend, args


def completed(prepared):
    run, backend, args = prepared
    segment.infer(run, args)
    segment.select(run)
    return run, backend, args


def test_raw_before_gates_and_cache_resume(prepared):
    run, backend, args = completed(prepared)
    assert backend.calls == 2
    index = read_json(run / "raw/index.json")
    text_entry = next(e for e in index["records"].values() if e["job"]["kind"] == "text")
    saved = segment.read_prediction(run / text_entry["file"])
    assert len(saved["candidates"]) == 3
    assert saved["candidates"][-1]["score"] == 0.01
    selected = read_json(run / "selected.json")
    assert selected["gate_counts"]["text_score_gate"] == 1
    old_index_hash = file_sha(run / "raw/index.json")
    segment.infer(run, args)
    assert backend.calls == 2
    assert file_sha(run / "raw/index.json") == old_index_hash
    assert (run / "selected.json").exists()


def test_interrupted_run_is_partial_and_resumable(prepared):
    run, backend, args = prepared
    backend.fail_points = True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        segment.infer(run, args)
    assert read_json(run / "raw/index.json")["status"] == "partial"
    with pytest.raises(ValueError, match="incomplete"):
        segment.select(run)
    backend.fail_points = False
    segment.infer(run, args)
    assert read_json(run / "raw/index.json")["status"] == "complete"
    assert backend.calls == 3
    segment.select(run)


def test_code_identity_change_refused_before_loading(prepared, monkeypatch):
    run, backend, args = completed(prepared)
    monkeypatch.setattr(segment, "code_identity", lambda: {"changed": "yes"})
    with pytest.raises(ValueError, match="different code"):
        segment.infer(run, args)
    assert backend.calls == 2


def test_runtime_change_requires_new_run(prepared):
    run, backend, args = completed(prepared)
    backend.identity = {"backend": "changed-runtime"}
    with pytest.raises(ValueError, match="different model/runtime"):
        segment.infer(run, args)


def test_corrupt_cache_not_reblessed(prepared):
    run, _, args = completed(prepared)
    index = read_json(run / "raw/index.json")
    entry = next(iter(index["records"].values()))
    envelope = read_json(run / entry["file"])
    envelope["payload"]["candidates"][0]["score"] = 0.123
    write_json(run / entry["file"], envelope)
    with pytest.raises(ValueError, match="checksum"):
        segment.infer(run, args)
    with pytest.raises(ValueError, match="checksum"):
        segment.select(run)


def test_embedded_identity_checked_even_without_prior_index(prepared):
    run, backend, args = completed(prepared)
    manifest = segment.manifest_for(run)
    job = segment.planned_jobs(manifest)[0]
    path, _ = segment.cached_call(run, Image.open(run / "original.png"), backend, manifest, job)
    envelope = read_json(path)
    envelope["payload"]["identity"]["request"]["phrase"] = "wrong request"
    envelope["payload_sha256"] = digest(envelope["payload"])
    write_json(path, envelope)
    with pytest.raises(ValueError, match="embedded identity"):
        segment.cached_call(run, Image.open(run / "original.png"), backend, manifest, job)


def test_raw_geometry_validated(prepared):
    run, backend, _ = completed(prepared)
    manifest = segment.manifest_for(run)
    job = segment.planned_jobs(manifest)[0]
    path, _ = segment.cached_call(run, Image.open(run / "original.png"), backend, manifest, job)
    envelope = read_json(path)
    envelope["payload"]["candidates"][0]["rle"] = encode(np.ones((2, 2), bool))
    envelope["payload_sha256"] = digest(envelope["payload"])
    write_json(path, envelope)
    with pytest.raises(ValueError, match="geometry"):
        segment.cached_call(run, Image.open(run / "original.png"), backend, manifest, job)


def test_selection_refinement_export_and_roundtrips(prepared):
    run, _, args = completed(prepared)
    args.limit = 3
    segment.refine(run, args)
    refined = read_json(run / "refined.json")
    assert len(refined["boundary_updates"]) <= 3
    assert all(m["status"] == "proposal" for m in refined["masks"])
    out = segment.export(run)
    coco = read_json(out / "coco.json")
    with ZipFile(out / "masks.zip") as archive:
        for obj in coco["annotations"]:
            image = (
                np.array(Image.open(io.BytesIO(archive.read(f"masks/{obj['instance_id']}.png"))))
                > 127
            )
            assert np.array_equal(image, decode(obj["segmentation"]))
            assert image.shape == (60, 80)
    assert segment.validate_result(run)["masks"] == len(coco["annotations"])
    assert (out / "index.html").exists()
    segment.export(run, omit_image=True)
    assert not (out / "original.png").exists()


def test_zero_refinement_does_not_load_model(prepared, monkeypatch):
    run, _, args = completed(prepared)
    args.limit = 0
    monkeypatch.setattr(
        segment, "backend_from", lambda *_: pytest.fail("Must not load model for zero targets")
    )
    segment.refine(run, args)
    assert read_json(run / "refined.json")["refinement_backend"] is None
    segment.validate_result(run)


def test_new_raw_evidence_archives_stale_outputs(prepared):
    run, _, args = completed(prepared)
    segment.export(run)
    manifest = segment.manifest_for(run)
    review = {
        "schema": "detailed-segmentation-review/1",
        "image_sha256": manifest["canonical_sha256"],
        "coordinate_space": "native",
        "reviewer": "fixture",
        "observations": [
            {
                "type": "missing",
                "points": [[25, 25]],
                "negative_points": [],
                "reason": "New test target",
                "uncertainty": "medium",
            }
        ],
    }
    args.review = run / "review-input.json"
    write_json(args.review, review)
    with pytest.raises(ValueError, match="explicitly"):
        segment.infer(run, args)
    args.confirm_review_points = True
    segment.infer(run, args)
    assert not (run / "selected.json").exists()
    assert not (run / "export").exists()
    assert list((run / "history").rglob("selected.json"))
    assert len(read_json(run / "raw/index.json")["records"]) == 3
    segment.select(run)
    segment.export(run)
    # Omitting --review on resume does not drop previously added review jobs.
    args.review = None
    segment.infer(run, args)
    assert len(read_json(run / "raw/index.json")["records"]) == 3
    assert (run / "selected.json").exists()


def test_stale_binding_and_partial_refinement_refused(prepared):
    run, _, _ = completed(prepared)
    index = read_json(run / "raw/index.json")
    index["changed"] = "new evidence"
    write_json(run / "raw/index.json", index)
    with pytest.raises(ValueError, match="stale"):
        segment.export(run)
    segment.select(run)
    write_json(run / "refinement-status.json", {"status": "partial"})
    with pytest.raises(ValueError, match="Refinement is incomplete"):
        segment.export(run)


def test_reselection_archives_refinement(prepared):
    run, _, args = completed(prepared)
    args.limit = 0
    segment.refine(run, args)
    assert (run / "refined.json").exists()
    segment.select(run)
    assert not (run / "refined.json").exists()
    assert list((run / "history").rglob("refined.json"))


def test_review_sheets_mapping_and_result_binding(prepared):
    run, _, _ = completed(prepared)
    segment.review_sheets(run, size=40, overlap=10, zoom=3)
    metadata = read_json(run / "review/sheets.json")
    first = metadata["sheets"][0]
    assert first["native_crop"] == [0, 0, 40, 40]
    assert first["header_height"] == 42
    assert first["overlay_rendered_origin"] == [120, 42]
    assert metadata["result_sha256"] == digest(read_json(run / "selected.json"))
    with Image.open(run / "review" / first["file"]) as panel:
        assert panel.size == (240, 162)
        assert panel.getpixel((0, 42)) == (80, 90, 100)


def test_damaged_geometry_and_unsafe_ids_rejected(prepared):
    run, _, _ = completed(prepared)
    result = read_json(run / "selected.json")
    original = copy.deepcopy(result)
    result["masks"][0]["bbox"][2] -= 1
    write_json(run / "selected.json", result)
    with pytest.raises(ValueError, match="Stale mask"):
        segment.validate_result(run)
    original["masks"][0]["id"] = "../../escape"
    write_json(run / "selected.json", original)
    with pytest.raises(ValueError, match="Invalid export instance ID"):
        segment.validate_result(run)


def test_canonical_image_change_rejected(prepared):
    run, _, _ = prepared
    Image.new("RGB", (80, 60), "red").save(run / "original.png")
    with pytest.raises(ValueError, match="Canonical image changed"):
        segment.manifest_for(run)


def test_prepare_never_overwrites_nonempty_directory(prepared):
    run, _, _ = prepared
    with pytest.raises(FileExistsError):
        segment.prepare(run / "original.png", run)


def test_budget_guard_before_model_load(prepared, monkeypatch):
    run, _, args = prepared
    manifest = read_json(run / "manifest.json")
    manifest["config"]["budget"]["max_candidates"] = 1
    write_json(run / "manifest.json", manifest)
    monkeypatch.setattr(
        segment, "backend_from", lambda *_: pytest.fail("Budget must be checked before loading")
    )
    with pytest.raises(ValueError, match="Budget exceeded"):
        segment.infer(run, args)


def test_synthetic_demo_end_to_end(tmp_path):
    run = tmp_path / "demo"
    segment.demo(run)
    assert read_json(run / "manifest.json")["synthetic_demo"] is True
    assert len(read_json(run / "selected.json")["masks"]) == 77
    assert read_json(run / "export/data.json")["synthetic"] is True
    assert segment.validate_result(run)["masks"] == 77


def test_relation_flags_mark_nested_parts_and_masks_outside_parents():
    def obj(x, y, w, h, label):
        mask = np.zeros((50, 50), bool)
        mask[y : y + h, x : x + w] = True
        return dict(
            label=label,
            rle=encode(mask),
            area=int(mask.sum()),
            bbox=[x, y, w, h],
            source={"kind": "text", "pass_name": "fine"},
            flags=["fragmented_boundary", "no_stable_agreeing_refinement"],
        )

    objects = [
        obj(0, 0, 30, 30, "ear"),
        obj(2, 2, 8, 8, "kernel"),  # inside the ear
        obj(1, 1, 20, 20, "kernel"),  # whole kernel containing the part above
        obj(35, 35, 8, 8, "kernel"),  # outside every ear
    ]
    config = {"label": "kernel", "whole_objects": [{"label": "ear", "phrase": "ear of corn"}]}
    segment.annotate_flags(objects, config)
    flags = [set(o["flags"]) for o in objects]
    assert "inside_larger_same_label_proposal" in flags[1]
    assert "inside_larger_same_label_proposal" not in flags[2]
    assert "outside_whole_object_proposals" in flags[3]
    assert "outside_whole_object_proposals" not in flags[1] | flags[2]
    # Geometry flags are recomputed; process flags from earlier stages persist.
    assert "fragmented_boundary" not in flags[1]
    assert "no_stable_agreeing_refinement" in flags[1]
    segment.annotate_flags(objects, {"label": "kernel", "whole_objects": []})
    assert "outside_whole_object_proposals" not in objects[3]["flags"]
