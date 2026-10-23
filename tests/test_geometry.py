import copy

import numpy as np
import pytest
from core import (
    DEFAULT_CONFIG,
    bbox,
    crop_iou,
    decode,
    dedupe,
    deepest,
    encode,
    fragment_fraction,
    iou,
    lift,
    mask_rows,
    record,
    stability,
    tile_starts,
    tiles,
    validate_config,
    validate_review,
)


@pytest.mark.parametrize("shape", [(1, 1), (3, 9), (11, 7), (500, 480)])
def test_rle_roundtrip(shape):
    mask = np.random.default_rng(1).random(shape) > 0.7
    assert np.array_equal(decode(encode(mask)), mask)


def test_last_row_column_and_one_pixel_boxes():
    mask = np.zeros((10, 11), bool)
    mask[-1, -1] = True
    assert bbox(mask) == [10, 9, 1, 1]
    assert crop_iou(mask[9:10, 10:11], bbox(mask), mask[9:10, 10:11], bbox(mask)) == 1
    mask[-2, -2] = True
    assert bbox(mask) == [9, 8, 2, 2]


@pytest.mark.parametrize("seed", range(30))
def test_cropped_iou_matches_full_array(seed):
    rng = np.random.default_rng(seed)
    masks = [rng.random((13, 19)) > 0.8 for _ in range(2)]
    crops = []
    for mask in masks:
        x, y, w, h = bbox(mask)
        crops.append(mask[y : y + h, x : x + w])
    assert crop_iou(crops[0], bbox(masks[0]), crops[1], bbox(masks[1])) == pytest.approx(
        iou(*masks)
    )


@pytest.mark.parametrize(
    "length,size,overlap",
    [(1, 120, 30), (120, 120, 0), (121, 120, 30), (480, 120, 30), (500, 200, 60)],
)
def test_tiling_covers_all_pixels(length, size, overlap):
    cover = np.zeros(length, bool)
    starts = tile_starts(length, size, overlap)
    assert len(starts) == len(set(starts))
    for start in starts:
        assert start >= 0
        cover[start : min(start + size, length)] = True
    assert cover.all()


def test_full_image_and_small_tiles():
    assert tiles(40, 50, None, 0) == [[0, 0, 40, 50]]
    assert tiles(40, 50, 120, 30) == [[0, 0, 40, 50]]
    with pytest.raises(ValueError):
        tile_starts(480, 100, 100)


def test_native_lift_and_internal_vs_true_edge():
    local = np.ones((5, 7), bool)
    full, cut = lift(local, [0, 0, 7, 5], (5, 7))
    assert full.all() and not cut
    full, cut = lift(local, [2, 3, 7, 5], (20, 20))
    assert cut and full[3:8, 2:9].all() and full.sum() == 35
    with pytest.raises(ValueError):
        lift(local, [2, 3, 6, 5], (20, 20))


def test_rows_recreate_exact_mask():
    mask = np.random.default_rng(55).random((21, 19)) > 0.6
    restored = np.zeros_like(mask)
    for y, x, w in mask_rows(mask):
        restored[y, x : x + w] = True
    assert np.array_equal(mask, restored)


def test_deepest_full_image_and_empty():
    assert deepest(np.ones((9, 9), bool)) == [4, 4]
    with pytest.raises(ValueError):
        deepest(np.zeros((3, 3), bool))


def test_fragment_connectivity_and_stability():
    assert fragment_fraction(np.eye(3, dtype=bool)) == 0
    mask = np.zeros((10, 10), bool)
    mask[0, 0] = mask[-1, -1] = True
    assert fragment_fraction(mask) == 0.5
    assert stability(np.array([-2, -0.5, 0.5, 2])) == pytest.approx(1 / 3)


def test_dedupe_keeps_nested_levels_and_records_provenance():
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    source = {"kind": "text", "pass_name": "a", "crop": [0, 0, 8, 8]}
    candidates = [
        record(mask, label, source, 0.9, candidate_id=str(i))
        for i, label in enumerate(["kernel", "kernel", "ear"])
    ]
    kept, suppressed = dedupe(candidates, 0.7)
    assert [k["label"] for k in kept] == ["kernel", "ear"]
    assert suppressed[0]["candidate_id"] == "1"
    assert suppressed[0]["kept_candidate_id"] == "0"
    assert kept[0]["matches"][0]["iou"] == 1


def test_config_rejects_invalid_overlap_and_area():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["passes"][1]["overlap"] = 200
    with pytest.raises(ValueError):
        validate_config(config)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["selection"]["max_area"] = 1
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "point", [[480, 20], [-1, 0], [20, 500], [float("nan"), 0], [float("inf"), 0]]
)
def test_review_rejects_invalid_coordinates(point):
    manifest = {"canonical_sha256": "abc", "width": 480, "height": 500}
    review = {
        "schema": "detailed-segmentation-review/1",
        "image_sha256": "abc",
        "coordinate_space": "native",
        "reviewer": "test",
        "observations": [
            {
                "type": "missing",
                "points": [point],
                "negative_points": [],
                "reason": "Test observation",
                "uncertainty": "medium",
            }
        ],
    }
    with pytest.raises(ValueError):
        validate_review(review, manifest)


def test_review_checks_image_identity_and_coordinate_system():
    manifest = {"canonical_sha256": "abc", "width": 480, "height": 500}
    review = {
        "schema": "detailed-segmentation-review/1",
        "image_sha256": "abc",
        "coordinate_space": "native",
        "reviewer": "test",
        "observations": [
            {
                "type": "missing",
                "points": [[479, 499]],
                "negative_points": [],
                "reason": "Test observation",
                "uncertainty": "medium",
            }
        ],
    }
    assert validate_review(review, manifest) is review
    review["coordinate_space"] = "rendered"
    with pytest.raises(ValueError):
        validate_review(review, manifest)
    review["coordinate_space"] = "native"
    review["image_sha256"] = "wrong"
    with pytest.raises(ValueError):
        validate_review(review, manifest)


def _square(size, x, y, w, h, label="kernel", candidate_id="c"):
    mask = np.zeros((size, size), bool)
    mask[y : y + h, x : x + w] = True
    source = {"kind": "text", "pass_name": "a", "crop": [0, 0, size, size]}
    return record(mask, label, source, 0.9, candidate_id=candidate_id)


def test_dedupe_containment_matches_similar_sized_copies_only():
    kernel = _square(40, 10, 10, 6, 6, candidate_id="kernel")
    leaky = _square(40, 10, 10, 6, 11, candidate_id="leaky")  # IoU 0.55, fully contained
    parent = _square(40, 5, 5, 20, 20, candidate_id="parent")  # 11x larger
    candidates = [kernel, leaky, parent]
    assert len(dedupe(candidates, 0.6)[0]) == 3
    kept, suppressed = dedupe(candidates, 0.6, containment=0.87, max_area_ratio=2.2)
    assert [k["candidate_id"] for k in kept] == ["kernel", "parent"]
    assert suppressed[0]["rule"] == "containment"
    assert suppressed[0]["contained_fraction"] == 1
    with pytest.raises(ValueError):
        dedupe(candidates, 0.6, containment=0.87, max_area_ratio=None)


def test_cropped_deepest_and_fragments_match_full_image_reference():
    from scipy import ndimage as ndi

    rng = np.random.default_rng(3)
    for _ in range(40):
        mask = np.zeros((37, 29), bool)
        y, x = rng.integers(0, 30), rng.integers(0, 22)
        mask[y : y + rng.integers(1, 8), x : x + rng.integers(1, 8)] = True
        mask &= rng.random(mask.shape) > 0.15
        if not mask.any():
            continue
        distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        ry, rx = np.unravel_index(distance.argmax(), mask.shape)
        assert deepest(mask) == [int(rx), int(ry)]
        labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
        counts = np.bincount(labels.ravel())[1:]
        assert fragment_fraction(mask) == pytest.approx(1 - counts.max() / mask.sum())


def test_config_validates_containment_settings():
    for name, value in [
        ("dedupe_containment", 0),
        ("dedupe_containment", 1.5),
        ("dedupe_containment_max_area_ratio", 0.5),
        ("dedupe_containment_max_area_ratio", True),
    ]:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["selection"][name] = value
        with pytest.raises(ValueError):
            validate_config(config)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["selection"]["dedupe_containment"] = None
    validate_config(config)
