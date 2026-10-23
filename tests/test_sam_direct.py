"""Direct SAM tool: coordinates, outputs and CLI plumbing with a fake model."""

import argparse
import json

import numpy as np
import pytest
import sam
from core import decode
from PIL import Image


class FakeSam(sam.Sam):
    """Geometry fixture only; never evidence of real-model quality."""

    def __init__(self):
        self.identity = {"backend": "fake"}
        self.calls = []

    def _find_local(self, crop, text, like, unlike):
        self.calls.append(("find", crop.size, text, like, unlike))
        a = np.zeros((crop.height, crop.width), bool)
        a[2:6, 3:9] = True
        b = np.zeros_like(a)
        b[10:12, 10:12] = True
        return [(a, 0.9, None), (b, 0.3, None)]

    def _click_local(self, crop, points, labels, box, multimask):
        self.calls.append(("click", crop.size, points, labels, box, multimask))
        masks = []
        for size, score in [(2, 0.5), (4, 0.95), (6, 0.7)]:
            m = np.zeros((crop.height, crop.width), bool)
            m[:size, :size] = True
            masks.append((m, score, 0.9))
        return masks if multimask else masks[1:2]

    def _points_local(self, crop, points):
        out = []
        for i, (x, y) in enumerate(points):
            m = np.zeros((crop.height, crop.width), bool)
            m[int(y) - 2 : int(y) + 3, int(x) - 2 : int(x) + 3] = True
            out.append((m, 0.95, 0.95, i))
            out.append((m.copy(), 0.96, 0.95, i))  # duplicate hypothesis
            out.append((m, 0.1, 0.95, i))  # low quality
        return out


@pytest.fixture
def image():
    return Image.new("RGB", (40, 30), (50, 60, 70))


def test_find_lifts_region_masks_to_native_coordinates(image):
    model = FakeSam()
    results = model.find(image, text="thing", like=[[12, 7, 4, 4]], region=[10, 5, 20, 20])
    assert len(results) == 1  # default threshold 0.5 drops the 0.3 mask
    assert results[0]["bbox"] == [13, 7, 6, 4]
    assert results[0]["mask"].shape == (30, 40)
    assert model.calls[0][3] == [[2, 2, 4, 4]]  # example box made region-local
    assert len(model.find(image, text="thing", threshold=0.2)) == 2


def test_example_boxes_use_normalized_centers():
    assert sam.normalized_center_box([10, 20, 30, 40], 100, 200) == [0.25, 0.2, 0.3, 0.2]


def test_click_returns_best_and_asks_for_alternatives_only_when_ambiguous(image):
    model = FakeSam()
    best = model.click(image, points=[[5, 5]])
    assert len(best) == 1 and best[0]["score"] == 0.95 and best[0]["area"] == 16
    assert model.calls[-1][-1] is True  # one click: three alternatives
    model.click(image, points=[[5, 5]], negatives=[[20, 20]])
    assert model.calls[-1][3] == [1, 0] and model.calls[-1][-1] is False
    every = model.click(image, points=[[5, 5]], all_hypotheses=True)
    assert [r["score"] for r in every] == [0.95, 0.7, 0.5]
    model.click(image, box=[1, 2, 3, 4], region=[1, 1, 20, 20])
    assert model.calls[-1][4] == [0, 1, 3, 4]


def test_everything_filters_and_removes_duplicates(image):
    results = FakeSam().everything(image, step=10, min_area=4)
    assert len(results) == 12  # one per grid point; duplicate and low-quality gone
    assert all(r["area"] == 25 or r["touches_region_edge"] is False for r in results)


@pytest.mark.parametrize(
    "call",
    [
        lambda m, i: m.find(i),
        lambda m, i: m.click(i),
        lambda m, i: m.click(i, points=[[50, 5]]),
        lambda m, i: m.find(i, text="x", region=[30, 20, 20, 20]),
        lambda m, i: m.find(i, text="x", like=[[0, 0, 5, 5]], region=[10, 10, 10, 10]),
    ],
)
def test_bad_requests_are_errors_not_clipped(image, call):
    with pytest.raises(ValueError):
        call(FakeSam(), image)


def test_cli_writes_image_json_overlay_and_never_overwrites(tmp_path, image, monkeypatch):
    source = tmp_path / "photo.png"
    image.save(source)
    monkeypatch.setattr(sam, "Sam", lambda **kwargs: FakeSam())
    out = tmp_path / "out"
    sam.main(["find", str(source), "--text", "thing", "--out", str(out), "--png"])
    sam.main(["find", str(source), "--text", "thing", "--out", str(out)])
    assert (out / "image.png").exists() and (out / "find-overlay.png").exists()
    assert (out / "find-masks" / "mask-001.png").exists()
    first = json.loads((out / "find.json").read_text())
    assert (out / "find-2.json").exists()
    assert first["masks"][0]["bbox"] == [3, 2, 6, 4]
    assert decode(first["masks"][0]["rle"]).shape == (30, 40)
    assert sam.load_masks(out / "find.json")[0]["mask"].sum() == 24
    other = tmp_path / "other.png"
    Image.new("RGB", (40, 30), (0, 0, 0)).save(other)
    with pytest.raises(ValueError):
        sam.main(["find", str(other), "--text", "thing", "--out", str(out)])


def test_batch_runs_several_requests_with_one_model(tmp_path, image, monkeypatch):
    source = tmp_path / "photo.png"
    image.save(source)
    made = []
    monkeypatch.setattr(sam, "Sam", lambda **kwargs: made.append(FakeSam()) or made[-1])
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        '{"command": "find", "text": "thing"}\n'
        '{"command": "click", "points": [[5, 5]], "name": "first-click"}\n'
        '{"command": "everything", "step": 10}\n'
    )
    sam.main(["batch", str(source), str(requests), "--out", str(tmp_path / "out")])
    assert len(made) == 1
    for name in ["find", "first-click", "everything"]:
        assert (tmp_path / "out" / f"{name}.json").exists()
    bad = tmp_path / "bad.json"
    bad.write_text('[{"command": "find", "name": "../x", "text": "a"}]')
    with pytest.raises(ValueError):
        sam.main(["batch", str(source), str(bad), "--out", str(tmp_path / "out2")])


def test_parsers_reject_malformed_coordinates():
    with pytest.raises(argparse.ArgumentTypeError):
        sam.parse_box("1,2,3")
    with pytest.raises(argparse.ArgumentTypeError):
        sam.parse_box("1,2,0,4")
    with pytest.raises(argparse.ArgumentTypeError):
        sam.parse_pair("1;2")


@pytest.mark.parametrize(
    "request_",
    [
        {"command": "find", "like": [1, 2, 3, 4]},  # one box instead of a list
        {"command": "find", "text": "a", "threshold": 2},
        {"command": "click", "points": [5, 5]},
        {"command": "click", "box": [1, 2, 0, 4]},
        {"command": "everything", "step": 0},
        {"command": "everything", "colour": "red"},
        {"command": "segment"},
        {"command": "find", "text": "a", "region": [0, 0, 5]},
    ],
)
def test_batch_requests_are_checked_before_loading_the_model(request_):
    with pytest.raises(ValueError):
        sam.check_request(request_)
