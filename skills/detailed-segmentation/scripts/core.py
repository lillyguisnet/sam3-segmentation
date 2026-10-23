"""Native-pixel mask geometry and content-addressed records. No model imports."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as coco
from scipy import ndimage as ndi

SCHEMA = "detailed-segmentation/1"
ALGORITHM = "native-pixel-v1"
DEFAULT_CONFIG = {
    "label": "kernel",
    "phrases": ["corn kernel", "kernel"],
    "whole_objects": [{"label": "ear", "phrase": "ear of corn"}],
    "passes": [
        {"name": "whole", "size": None, "overlap": 0, "grid_step": None},
        {"name": "coarse", "size": 200, "overlap": 60, "grid_step": 6},
        {"name": "fine", "size": 120, "overlap": 30, "grid_step": None},
    ],
    "selection": {
        "text_score_min": 0.20,
        "point_quality_min": None,
        "point_stability_min": 0.83,
        "min_area": 5,
        "max_area": 650,
        "dedupe_iou": 0.52,
        "exclude_internal_crop_edges": True,
    },
    "refinement": {
        "crop_size": 120,
        "quality_min": 0.70,
        "stability_min": 0.82,
        "agreement_min": 0.50,
        "supported_agreement_min": 0.68,
        "area_ratio_min": 0.50,
        "area_ratio_max": 1.65,
    },
    "budget": {"max_jobs": 256, "max_grid_points": 25000, "max_candidates": 100000},
    "seed": 0,
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        if str(path).endswith(".gz"):
            with gzip.open(temp, "wt", encoding="utf-8") as stream:
                stream.write(canonical_json(value))
        else:
            temp.write_text(canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def encode(mask):
    result = coco.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    result["counts"] = result["counts"].decode("ascii")
    return result


def decode(rle):
    return coco.decode(rle).astype(bool)


def bbox(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def iou(a, b):
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else 0.0


def crop_iou(a, box_a, b, box_b):
    """Intersect pixel-tight crops, including the last occupied row and column."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    intersection = int(
        np.count_nonzero(
            a[y0 - ay : y1 - ay, x0 - ax : x1 - ax] & b[y0 - by : y1 - by, x0 - bx : x1 - bx]
        )
    )
    union = int(a.sum()) + int(b.sum()) - intersection
    return intersection / union if union else 0.0


def tile_starts(length, size, overlap):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Tile size must be positive and 0 <= overlap < size")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def tiles(width, height, size, overlap):
    if size is None:
        return [[0, 0, width, height]]
    return [
        [x, y, min(size, width), min(size, height)]
        for y in tile_starts(height, size, overlap)
        for x in tile_starts(width, size, overlap)
    ]


def grid_points(width, height, step):
    if step <= 0:
        raise ValueError("Grid step must be positive")
    return [
        [float(x), float(y)]
        for y in np.arange(step / 2, height, step)
        for x in np.arange(step / 2, width, step)
    ]


def stability(logits, delta=1.0):
    return float(np.count_nonzero(logits > delta) / max(1, np.count_nonzero(logits > -delta)))


def record(mask, label, source, score=None, **extra):
    return dict(
        label=label,
        rle=encode(mask),
        area=int(mask.sum()),
        bbox=bbox(mask),
        source=source,
        score=score,
        status="proposal",
        **extra,
    )


def lift(local, crop, shape):
    x, y, width, height = crop
    full_height, full_width = shape
    if (
        local.shape != (height, width)
        or x < 0
        or y < 0
        or x + width > full_width
        or y + height > full_height
    ):
        raise ValueError("Crop/mask geometry mismatch")
    result = np.zeros(shape, dtype=bool)
    result[y : y + height, x : x + width] = local
    cut = (
        (x > 0 and local[:, 0].any())
        or (y > 0 and local[0].any())
        or (x + width < full_width and local[:, -1].any())
        or (y + height < full_height and local[-1].any())
    )
    return result, bool(cut)


def deepest(mask):
    if not mask.any():
        raise ValueError("Cannot prompt an empty mask")
    # Explicit background padding gives a meaningful center even for a full mask.
    distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(distance.argmax(), mask.shape)
    return [int(x), int(y)]


def fragment_fraction(mask):
    components, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    counts = np.bincount(components.ravel())[1:]
    return 1 - float(counts.max(initial=0)) / max(1, int(mask.sum()))


def dedupe(candidates, threshold):
    """Greedy same-label suppression; never makes parts/wholes disjoint."""
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    kept, cropped, suppressions = [], [], []
    for candidate in candidates:
        if not candidate["area"]:
            continue
        mask = decode(candidate["rle"])
        x, y, w, h = candidate["bbox"]
        local = mask[y : y + h, x : x + w]
        for j, previous in enumerate(kept):
            if candidate["label"] != previous["label"]:
                continue
            overlap = crop_iou(local, candidate["bbox"], cropped[j], previous["bbox"])
            if overlap >= threshold:
                match = dict(
                    candidate_id=candidate["candidate_id"], source=candidate["source"], iou=overlap
                )
                previous.setdefault("matches", []).append(match)
                suppressions.append(dict(**match, kept_candidate_id=previous["candidate_id"]))
                break
        else:
            kept.append(dict(candidate))
            cropped.append(local)
    return kept, suppressions


def mask_rows(mask):
    rows = []
    for y, row in enumerate(mask):
        difference = np.diff(np.r_[False, row, False].astype(np.int8))
        rows.extend(
            [int(y), int(a), int(b - a)]
            for a, b in zip(
                np.flatnonzero(difference == 1), np.flatnonzero(difference == -1), strict=True
            )
        )
    return rows


def canonical_image(path, orientation):
    with Image.open(path) as image:
        image.load()
        if orientation == "exif":
            image = ImageOps.exif_transpose(image)
        elif orientation != "raw":
            raise ValueError("Orientation must be raw or exif")
        return image.convert("RGB")


def require_keys(value, required, optional=(), context="object"):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise ValueError(f"Invalid keys or container for {context}; required: {sorted(required)}")


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and np.isfinite(value)


def probability(value, nullable=False):
    return (nullable and value is None) or (finite(value) and 0 <= value <= 1)


def validate_config(config):
    require_keys(config, DEFAULT_CONFIG, context="configuration")
    if (
        not nonempty(config["label"])
        or not isinstance(config["phrases"], list)
        or not config["phrases"]
    ):
        raise ValueError("Config needs a nonempty label and phrase list")
    if not all(nonempty(p) for p in config["phrases"]):
        raise ValueError("Phrases must be nonempty strings")
    if not isinstance(config["whole_objects"], list):
        raise ValueError("whole_objects must be a list")
    for whole in config["whole_objects"]:
        require_keys(whole, ["label", "phrase"], context="whole object")
        if not nonempty(whole["label"]) or not nonempty(whole["phrase"]):
            raise ValueError("Whole-object labels and phrases must be nonempty strings")
    if not isinstance(config["passes"], list) or not config["passes"]:
        raise ValueError("At least one pass is required")
    for p in config["passes"]:
        require_keys(p, ["name", "size", "overlap", "grid_step"], context="pass")
        if not nonempty(p["name"]) or not integer(p["overlap"]):
            raise ValueError("Pass name and integer overlap are required")
        if p["size"] is not None:
            if not integer(p["size"], 1):
                raise ValueError("Crop size must be a positive integer or null")
            tile_starts(1, p["size"], p["overlap"])
        elif p["overlap"] != 0:
            raise ValueError("Whole-image overlap must be zero")
        if p["grid_step"] is not None and not integer(p["grid_step"], 1):
            raise ValueError("Grid step must be a positive integer or null")
    names = [p["name"] for p in config["passes"]]
    if len(names) != len(set(names)):
        raise ValueError("Pass names must be unique")
    s = config["selection"]
    require_keys(s, DEFAULT_CONFIG["selection"], context="selection")
    if not probability(s["dedupe_iou"]) or s["dedupe_iou"] == 0 or not integer(s["min_area"]):
        raise ValueError("Invalid selection IoU or minimum area")
    if s["max_area"] is not None and (not integer(s["max_area"]) or s["max_area"] < s["min_area"]):
        raise ValueError("Maximum area must be an integer >= min_area, or null")
    for name in ["text_score_min", "point_quality_min", "point_stability_min"]:
        if not probability(s[name], nullable=True):
            raise ValueError(f"{name} must be finite in [0,1], or null")
    if type(s["exclude_internal_crop_edges"]) is not bool:
        raise ValueError("exclude_internal_crop_edges must be boolean")
    r = config["refinement"]
    require_keys(r, DEFAULT_CONFIG["refinement"], context="refinement")
    if not integer(r["crop_size"], 2):
        raise ValueError("Refinement crop size must be an integer >= 2")
    for name in ["quality_min", "stability_min", "agreement_min", "supported_agreement_min"]:
        if not probability(r[name]):
            raise ValueError(f"Refinement {name} must be finite in [0,1]")
    if r["supported_agreement_min"] < r["agreement_min"]:
        raise ValueError("Supported agreement cannot be less strict than basic agreement")
    if not all(finite(r[name]) and r[name] > 0 for name in ["area_ratio_min", "area_ratio_max"]):
        raise ValueError("Refinement area ratios must be finite and positive")
    if r["area_ratio_min"] > r["area_ratio_max"]:
        raise ValueError("Refinement area ratio bounds are reversed")
    require_keys(config["budget"], DEFAULT_CONFIG["budget"], context="budget")
    if not all(integer(value, 1) for value in config["budget"].values()):
        raise ValueError("Budgets must be positive integers")
    if not integer(config["seed"]) or config["seed"] >= 2**32:
        raise ValueError("Seed must be an integer in [0,2**32)")
    return config


def validate_review(review, manifest):
    require_keys(
        review,
        ["schema", "image_sha256", "coordinate_space", "reviewer", "observations"],
        context="review",
    )
    if review["schema"] != "detailed-segmentation-review/1":
        raise ValueError("Wrong review schema")
    if review["image_sha256"] != manifest["canonical_sha256"]:
        raise ValueError("Review refers to a different canonical image")
    if review["coordinate_space"] != "native":
        raise ValueError("Convert review coordinates to native image pixels first")
    if not nonempty(review["reviewer"]) or not isinstance(review["observations"], list):
        raise ValueError("Reviewer must be identified and observations must be a list")
    width, height = manifest["width"], manifest["height"]
    for observation in review["observations"]:
        require_keys(
            observation,
            ["type", "points", "negative_points", "reason", "uncertainty"],
            optional=["mask_id"],
            context="review observation",
        )
        if observation["type"] not in {"missing", "merged", "fragmented", "leakage", "uncertain"}:
            raise ValueError("Unknown review observation type")
        if not nonempty(observation["reason"]) or observation["uncertainty"] not in {
            "low",
            "medium",
            "high",
            "unresolved",
        }:
            raise ValueError("Observation needs a reason and explicit uncertainty")
        if not isinstance(observation["points"], list) or not isinstance(
            observation["negative_points"], list
        ):
            raise ValueError("Review point containers must be lists")
        if not observation["points"] and observation["type"] != "uncertain":
            raise ValueError("Actionable observations need a positive target point")
        if "mask_id" in observation and not nonempty(observation["mask_id"]):
            raise ValueError("mask_id must be a nonempty string")
        for point in observation["points"] + observation["negative_points"]:
            if not isinstance(point, list) or len(point) != 2 or not all(finite(v) for v in point):
                raise ValueError("Review points must be finite [x,y] coordinates")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError(f"Review point outside native image: {point}")
    return review
