#!/usr/bin/env python3
"""Portable, resumable detailed instance-mask workflow. See --help."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import io
import json
import platform
import random
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from core import (
    ALGORITHM,
    DEFAULT_CONFIG,
    SCHEMA,
    bbox,
    canonical_image,
    crop_intersection,
    decode,
    dedupe,
    deepest,
    digest,
    encode,
    file_sha,
    fragment_fraction,
    grid_points,
    iou,
    lift,
    local_crop,
    mask_rows,
    read_json,
    record,
    tiles,
    validate_config,
    validate_review,
    write_json,
)
from filelock import FileLock, Timeout
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def merge_config(base, changes):
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        merge_config(config, read_json(path))
    return validate_config(config)


def manifest_for(run):
    manifest = read_json(run / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported run schema")
    if file_sha(run / "original.png") != manifest["canonical_sha256"]:
        raise ValueError("Canonical image changed; prepare a new run instead")
    return manifest


def prepare(image_path, out, config=None, orientation="raw", selection=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Output is not empty. Use a new run directory; old evidence is never overwritten."
        )
    config = validate_config(config or load_config())
    image = canonical_image(image_path, orientation)
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "original.png")
    manifest = dict(
        schema=SCHEMA,
        algorithm=ALGORITHM,
        created_at=now(),
        status="in_progress",
        source_name=Path(image_path).name,
        source_sha256=file_sha(image_path),
        canonical_sha256=file_sha(out / "original.png"),
        width=image.width,
        height=image.height,
        orientation=orientation,
        selection=selection,
        config=config,
        warning="All masks are unaccepted proposals. Counts and model signals are not accuracy.",
    )
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {image.width}x{image.height} native pixels in {out}")
    return manifest


def planned_jobs(manifest):
    width, height = manifest["width"], manifest["height"]
    config = manifest["config"]
    result = []
    for stage in config["passes"]:
        for crop in tiles(width, height, stage["size"], stage["overlap"]):
            for phrase in config["phrases"]:
                result.append(
                    dict(
                        kind="text",
                        pass_name=stage["name"],
                        crop=crop,
                        label=config["label"],
                        phrase=phrase,
                    )
                )
            if stage.get("grid_step"):
                prompts = [
                    dict(points=[p], labels=[1])
                    for p in grid_points(crop[2], crop[3], stage["grid_step"])
                ]
                result.append(
                    dict(
                        kind="point",
                        pass_name=stage["name"],
                        crop=crop,
                        label=config["label"],
                        prompts=prompts,
                    )
                )
    for whole in config.get("whole_objects", []):
        result.append(
            dict(
                kind="text",
                pass_name="whole-object",
                crop=[0, 0, width, height],
                label=whole["label"],
                phrase=whole["phrase"],
            )
        )
    return result


def review_jobs(review, manifest):
    validate_review(review, manifest)
    width, height = manifest["width"], manifest["height"]
    size = manifest["config"]["refinement"]["crop_size"]
    result = []
    for observation in review.get("observations", []):
        for px, py in observation.get("points", []):
            w, h = min(size, width), min(size, height)
            x, y = max(0, min(width - w, int(px - w / 2))), max(0, min(height - h, int(py - h / 2)))
            negatives = [
                [nx - x, ny - y]
                for nx, ny in observation.get("negative_points", [])
                if x <= nx < x + w and y <= ny < y + h
            ]
            prompt = dict(points=[[px - x, py - y]] + negatives, labels=[1] + [0] * len(negatives))
            result.append(
                dict(
                    kind="point",
                    pass_name="review-" + digest(review)[:12],
                    crop=[x, y, w, h],
                    label=manifest["config"]["label"],
                    prompts=[prompt],
                    review_observation=observation,
                )
            )
    return result


def budget_check(jobs, config):
    points = sum(len(j.get("prompts", [])) for j in jobs)
    # Upper-bound estimate for the pinned detector's 200 query slots.
    estimated = sum(200 if j["kind"] == "text" else 3 * len(j["prompts"]) for j in jobs)
    usage = dict(max_jobs=len(jobs), max_grid_points=points, max_candidates=estimated)
    errors = [
        f"{key}: planned {value:,} exceeds budget {config['budget'][key]:,}"
        for key, value in usage.items()
        if value > config["budget"][key]
    ]
    return dict(
        jobs=len(jobs),
        point_prompts=points,
        candidate_upper_bound=estimated,
        budget=config["budget"],
        errors=errors,
    )


def backend_from(args, manifest):
    from sam3_backend import Sam3Backend

    return Sam3Backend(
        device=args.device,
        checkpoint=args.checkpoint,
        precision=args.precision,
        batch_size=args.batch_size,
        seed=manifest["config"]["seed"],
    )


def tools_identity():
    return dict(
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pillow", "pycocotools", "filelock"]
        },
    )


def code_identity():
    return {
        p.name: file_sha(p)
        for p in [HERE / "core.py", HERE / "sam3_backend.py", HERE / "segment.py"]
    }


def archive_downstream(run, names):
    existing = [run / name for name in names if (run / name).exists()]
    if not existing:
        return
    destination = (
        run
        / "history"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    )
    destination.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), destination / path.name)


def workflow_identity(manifest):
    return dict(
        algorithm=ALGORITHM,
        code=code_identity(),
        image=manifest["canonical_sha256"],
        config=digest(manifest["config"]),
    )


def request_identity(manifest, backend_identity, job):
    return dict(**workflow_identity(manifest), backend=backend_identity, request=job)


def read_prediction(path, identity=None, expected_sha=None):
    if expected_sha is not None and file_sha(path) != expected_sha:
        raise ValueError("Raw prediction file checksum mismatch")
    envelope = read_json(path)
    if (
        set(envelope) != {"payload_sha256", "payload"}
        or digest(envelope["payload"]) != envelope["payload_sha256"]
    ):
        raise ValueError("Raw cache payload checksum mismatch")
    saved = envelope["payload"]
    if identity is not None and saved["identity"] != identity:
        raise ValueError("Raw cache embedded identity mismatch")
    job = saved["identity"]["request"]
    _, _, width, height = job["crop"]
    if not isinstance(saved["candidates"], list):
        raise ValueError("Raw candidates must be a list")
    ids = set()
    for candidate in saved["candidates"]:
        mask = decode(candidate["rle"])
        if mask.shape != (height, width) or not np.isfinite(candidate["score"]):
            raise ValueError("Raw candidate geometry or score is invalid")
        if candidate["candidate_id"] in ids:
            raise ValueError("Duplicate raw candidate identity")
        ids.add(candidate["candidate_id"])
        if job["kind"] == "point":
            if type(candidate["prompt_index"]) is not int or not 0 <= candidate[
                "prompt_index"
            ] < len(job["prompts"]):
                raise ValueError("Invalid independent prompt index")
            if not np.isfinite(candidate["stability"]) or not 0 <= candidate["stability"] <= 1:
                raise ValueError("Invalid mask stability")
    return saved


def cached_call(run, image, backend, manifest, job, folder="raw", expected_sha=None):
    identity = request_identity(manifest, backend.identity, job)
    key = digest(identity)
    path = run / folder / (key + ".json.gz")
    if path.exists():
        return path, read_prediction(path, identity, expected_sha)
    if expected_sha is not None:
        raise ValueError(
            "A previously indexed raw cache file is missing; recover it or use a new run"
        )
    x, y, w, h = job["crop"]
    crop = image.crop((x, y, x + w, y + h))
    started = time.perf_counter()
    candidates = (
        backend.text(crop, job["phrase"])
        if job["kind"] == "text"
        else backend.points(crop, job["prompts"])
    )
    for i, candidate in enumerate(candidates):
        candidate["candidate_id"] = key[:20] + f"-{i:05d}"
    result = dict(
        identity=identity,
        created_at=now(),
        seconds=time.perf_counter() - started,
        candidates=candidates,
    )
    write_json(path, dict(payload_sha256=digest(result), payload=result))
    return path, read_prediction(path, identity)


def infer(run, args):
    manifest = manifest_for(run)
    index_path = run / "raw" / "index.json"
    existing = read_json(index_path) if index_path.exists() else None
    workflow = workflow_identity(manifest)
    if existing is not None and existing.get("workflow") != workflow:
        raise ValueError(
            "Run contains a different code/config/image identity. Use a new run; mixed code generations are forbidden."
        )
    jobs = {digest(job): job for job in planned_jobs(manifest)}
    if existing:
        jobs.update({key: entry["job"] for key, entry in existing["records"].items()})
    if args.review:
        if not args.confirm_review_points:
            raise ValueError(
                "Inspect the review points first, then explicitly pass --confirm-review-points"
            )
        jobs.update({digest(job): job for job in review_jobs(read_json(args.review), manifest)})
    plan = budget_check(list(jobs.values()), manifest["config"])
    print(json.dumps(plan, indent=2))
    if plan["errors"]:
        raise ValueError(
            "Budget exceeded before loading weights. Prepare a new run with an explicit larger budget."
        )
    backend = backend_from(args, manifest)
    image = Image.open(run / "original.png").convert("RGB")
    index = existing or dict(
        workflow=workflow, backend=backend.identity, records={}, status="partial"
    )
    if index["backend"] != backend.identity:
        raise ValueError(
            "This run already contains a different model/runtime identity. Use a new run."
        )
    for number, (request_id, job) in enumerate(jobs.items(), 1):
        previous = index["records"].get(request_id)
        path, result = cached_call(
            run,
            image,
            backend,
            manifest,
            job,
            expected_sha=previous["sha256"] if previous else None,
        )
        entry = dict(
            file=str(path.relative_to(run)),
            sha256=file_sha(path),
            count=len(result["candidates"]),
            job=job,
        )
        if entry != previous:
            archive_downstream(
                run,
                [
                    "selected.json",
                    "refined.json",
                    "refinement-status.json",
                    "review",
                    "export",
                    "threshold-audit.json",
                ],
            )
            index["records"][request_id] = entry
            index["status"] = "partial"
            write_json(index_path, index)
        print(
            f"{number}/{len(jobs)} {job['pass_name']} {job['crop']} {job['kind']}: {len(result['candidates'])} hypotheses",
            flush=True,
        )
        if (
            sum(r["count"] for r in index["records"].values())
            > manifest["config"]["budget"]["max_candidates"]
        ):
            raise ValueError(
                "Actual candidate budget exceeded. Saved hypotheses are intact; nothing was silently truncated."
            )
    index["status"] = "complete"
    write_json(index_path, index)


def materialize(run, manifest, selection):
    index = read_json(run / "raw" / "index.json")
    if index.get("status") != "complete":
        raise ValueError("Inference is incomplete. Resume it before selecting a result.")
    if index.get("workflow") != workflow_identity(manifest):
        raise ValueError(
            "Raw index code/config/image identity differs from this workflow; use a new run"
        )
    candidates, rejected = [], Counter()
    for entry in index["records"].values():
        path = run / entry["file"]
        identity = request_identity(manifest, index["backend"], entry["job"])
        saved = read_prediction(path, identity, entry["sha256"])
        job = saved["identity"]["request"]
        for item in saved["candidates"]:
            local = decode(item["rle"])
            if not local.any():
                rejected["empty"] += 1
                continue
            area = int(local.sum())
            small_label = job["label"] == manifest["config"]["label"]
            if small_label and (
                area < selection["min_area"]
                or (selection["max_area"] is not None and area > selection["max_area"])
            ):
                rejected["area_gate"] += 1
                continue
            if job["kind"] == "text":
                threshold = selection["text_score_min"]
                if threshold is not None and item["score"] < threshold:
                    rejected["text_score_gate"] += 1
                    continue
            else:
                quality = selection["point_quality_min"]
                stable = selection["point_stability_min"]
                if quality is not None and item["score"] < quality:
                    rejected["point_quality_gate"] += 1
                    continue
                if stable is not None and item["stability"] < stable:
                    rejected["point_stability_gate"] += 1
                    continue
                prompt = job["prompts"][item["prompt_index"]]
                if not all(
                    local[min(int(py), local.shape[0] - 1), min(int(px), local.shape[1] - 1)]
                    for (px, py), label in zip(prompt["points"], prompt["labels"], strict=True)
                    if label == 1
                ):
                    rejected["positive_point_not_in_mask"] += 1
                    continue
            full, cut = lift(local, job["crop"], (manifest["height"], manifest["width"]))
            if cut and selection["exclude_internal_crop_edges"]:
                rejected["internal_crop_edge"] += 1
                continue
            source = dict(
                kind=job["kind"],
                pass_name=job["pass_name"],
                crop=job["crop"],
                raw_file=entry["file"],
                semantic_status="text-conditioned proposal"
                if job["kind"] == "text"
                else "unverified point candidate",
            )
            if "phrase" in job:
                source["phrase"] = job["phrase"]
            else:
                source["prompt"] = job["prompts"][item["prompt_index"]]
            candidate = record(
                full,
                job["label"],
                source,
                item["score"],
                candidate_id=item["candidate_id"],
                stability=item.get("stability"),
                crop_edge_cut=cut,
            )
            candidates.append(candidate)
    # Text and point-head scores are not pooled as interchangeable probabilities.
    # Prefer text-conditioned fine crops; use point proposals to recover gaps.
    candidates.sort(
        key=lambda r: (
            r["source"]["kind"] != "text",
            r["source"]["crop"][2] * r["source"]["crop"][3],
            -r["score"],
        )
    )
    return candidates, dict(rejected)


# Flags derived from the current geometry. They are recomputed at every stage so a
# boundary update cannot leave a stale warning (or hide a new one).
COMPUTED_FLAGS = {
    "fragmented_boundary",
    "tiny_instance",
    "point_semantics_unverified",
    "single_pass_only",
    "inside_larger_same_label_proposal",
    "outside_whole_object_proposals",
}
NESTED_FLAG_FRACTION = 0.87
OUTSIDE_FLAG_FRACTION = 0.15


def relation_flags(objects, config):
    """Cheap identity warnings for review; never used to delete a proposal.

    - A mask mostly inside a larger same-label proposal is often a part (a wing,
      a lid, half a kernel) rather than another instance.
    - A main-label mask largely outside every whole-object proposal (for corn,
      outside every ear) is often background. Skipped when no parents exist.
    """
    flags = [set() for _ in objects]
    locals_ = [local_crop(decode(obj["rle"]), obj["bbox"]) for obj in objects]
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j or a["label"] != b["label"] or b["area"] <= a["area"]:
                continue
            inter = crop_intersection(locals_[i], a["bbox"], locals_[j], b["bbox"])
            if inter >= NESTED_FLAG_FRACTION * a["area"]:
                flags[i].add("inside_larger_same_label_proposal")
                break
    if config is not None:
        parents = {whole["label"] for whole in config.get("whole_objects", [])}
        parent_objects = [k for k, obj in enumerate(objects) if obj["label"] in parents]
        if parent_objects:
            for i, obj in enumerate(objects):
                if obj["label"] != config["label"]:
                    continue
                covered = np.zeros_like(locals_[i])
                x, y, w, h = obj["bbox"]
                for k in parent_objects:
                    px, py, pw, ph = objects[k]["bbox"]
                    x0, y0 = max(x, px), max(y, py)
                    x1, y1 = min(x + w, px + pw), min(y + h, py + ph)
                    if x0 < x1 and y0 < y1:
                        covered[y0 - y : y1 - y, x0 - x : x1 - x] |= locals_[k][
                            y0 - py : y1 - py, x0 - px : x1 - px
                        ]
                outside = int(np.count_nonzero(locals_[i] & ~covered))
                if outside > OUTSIDE_FLAG_FRACTION * obj["area"]:
                    flags[i].add("outside_whole_object_proposals")
    return flags


def annotate_flags(objects, config=None):
    related = relation_flags(objects, config)
    for obj, extra in zip(objects, related, strict=True):
        mask = decode(obj["rle"])
        flags = (set(obj.get("flags", [])) - COMPUTED_FLAGS) | extra
        if fragment_fraction(mask) > 0.04:
            flags.add("fragmented_boundary")
        if obj["area"] < 30:
            flags.add("tiny_instance")
        if obj.get("source", {}).get("kind") == "point":
            flags.add("point_semantics_unverified")
        passes = {obj["source"].get("pass_name")} | {
            m["source"].get("pass_name") for m in obj.get("matches", [])
        }
        obj["multiscale_support"] = len(passes - {"whole-object"}) > 1
        if not obj["multiscale_support"]:
            flags.add("single_pass_only")
        obj["flags"] = sorted(flags)
        obj["status"] = "proposal"


def assign_ids(objects):
    centers = {id(r): deepest(decode(r["rle"])) for r in objects}
    objects.sort(key=lambda r: (r["label"], centers[id(r)][1], centers[id(r)][0]))
    for i, obj in enumerate(objects, 1):
        obj["id"] = f"mask-{i:05d}"
    return objects


def select(run, override=None):
    manifest = manifest_for(run)
    settings = copy.deepcopy(manifest["config"]["selection"])
    if override:
        merge_config(settings, read_json(override))
    check = copy.deepcopy(manifest["config"])
    check["selection"] = settings
    validate_config(check)
    candidates, rejected = materialize(run, manifest, settings)
    objects, suppressed = dedupe(
        candidates,
        settings["dedupe_iou"],
        settings["dedupe_containment"],
        settings["dedupe_containment_max_area_ratio"],
    )
    annotate_flags(objects, manifest["config"])
    assign_ids(objects)
    result = dict(
        schema=SCHEMA,
        canonical_sha256=manifest["canonical_sha256"],
        status="in_progress",
        selection=settings,
        selection_runtime=tools_identity(),
        masks=objects,
        gate_counts=rejected,
        suppressions=suppressed,
        raw_index_sha256=file_sha(run / "raw" / "index.json"),
        note="Suppression and score gates are heuristics, not semantic verification.",
    )
    archive_downstream(
        run, ["selected.json", "refined.json", "refinement-status.json", "review", "export"]
    )
    write_json(run / "selected.json", result)
    print(
        f"Selected {len(objects)} proposals; {len(suppressed)} suppressed duplicates. Raw hypotheses remain in raw/."
    )
    return result


def current(run):
    path = run / "refined.json" if (run / "refined.json").exists() else run / "selected.json"
    result = read_json(path)
    manifest = manifest_for(run)
    if result["canonical_sha256"] != manifest["canonical_sha256"]:
        raise ValueError("Selected masks refer to a different image")
    if not manifest.get("synthetic_demo"):
        selected = read_json(run / "selected.json")
        index_path = run / "raw" / "index.json"
        if read_json(index_path).get("status") != "complete" or selected.get(
            "raw_index_sha256"
        ) != file_sha(index_path):
            raise ValueError(
                "Selection is stale or inference incomplete; rerun select before exporting"
            )
        status_path = run / "refinement-status.json"
        if status_path.exists() and read_json(status_path).get("status") != "complete":
            raise ValueError("Refinement is incomplete; resume it before exporting")
        if path.name == "refined.json" and result.get("selection_sha256") != file_sha(
            run / "selected.json"
        ):
            raise ValueError("Refinement is stale; rerun it after the current selection")
    return result


def refine(run, args):
    manifest = manifest_for(run)
    selected = read_json(run / "selected.json")
    if selected["canonical_sha256"] != manifest["canonical_sha256"]:
        raise ValueError("Selection refers to a different canonical image")
    index_path = run / "raw" / "index.json"
    if read_json(index_path).get("status") != "complete" or selected.get(
        "raw_index_sha256"
    ) != file_sha(index_path):
        raise ValueError("Selection is stale or inference incomplete; rerun select first")
    objects = copy.deepcopy(selected["masks"])
    target = manifest["config"]["label"]
    settings = manifest["config"]["refinement"]
    width, height = manifest["width"], manifest["height"]
    size = settings["crop_size"]
    # Decode one mask at a time: keeping every full-image mask alive costs
    # gigabytes on a phone-sized photo with thousands of proposals.
    centers = [deepest(decode(obj["rle"])) for obj in objects]
    groups = {}
    attempted = 0
    for i, obj in enumerate(objects):
        if obj["label"] != target or (args.limit is not None and attempted >= args.limit):
            continue
        x, y, w, h = obj["bbox"]
        cx, cy = centers[i]
        options = [
            crop
            for crop in tiles(width, height, size, min(size - 1, max(1, size // 3)))
            if crop[0] <= x
            and crop[1] <= y
            and crop[0] + crop[2] >= x + w
            and crop[1] + crop[3] >= y + h
        ]
        if not options:
            obj.setdefault("flags", []).append("no_complete_refinement_crop")
            continue
        crop = min(options, key=lambda c: (cx - c[0] - c[2] / 2) ** 2 + (cy - c[1] - c[3] / 2) ** 2)
        X, Y, cw, ch = crop
        own = decode(obj["rle"])
        negatives = []
        for j in sorted(
            range(len(objects)), key=lambda j: np.linalg.norm(np.array(centers[j]) - centers[i])
        ):
            qx, qy = centers[j]
            if j != i and objects[j]["label"] == target and X <= qx < X + cw and Y <= qy < Y + ch:
                if np.linalg.norm(np.array(centers[j]) - centers[i]) >= 5 and not own[qy, qx]:
                    negatives.append([qx - X, qy - Y])
                    if len(negatives) == 3:
                        break
        if not negatives:
            obj.setdefault("flags", []).append("no_trustworthy_neighbor_negatives")
            continue
        prompt = dict(points=[[cx - X, cy - Y]] + negatives, labels=[1] + [0] * len(negatives))
        groups.setdefault(tuple(crop), []).append((i, prompt))
        attempted += 1
    budget = manifest["config"]["budget"]
    if (
        attempted > budget["max_grid_points"]
        or len(groups) > budget["max_jobs"]
        or 3 * attempted > budget["max_candidates"]
    ):
        raise ValueError("Refinement budget exceeded before loading the model")
    archive_downstream(run, ["refined.json", "refinement-status.json", "export"])
    write_json(
        run / "refinement-status.json",
        dict(status="partial", selection_sha256=file_sha(run / "selected.json")),
    )
    backend = backend_from(args, manifest) if groups else None
    image = Image.open(run / "original.png").convert("RGB")
    updates = []
    for crop, group in groups.items():
        # Include initial masks in the request cache identity; changed selections cannot reuse stale updates.
        job = dict(
            kind="point",
            pass_name="refinement",
            label=target,
            crop=list(crop),
            prompts=[p for _, p in group],
            initial_masks=[digest(objects[i]["rle"]) for i, _ in group],
        )
        path, saved = cached_call(run, image, backend, manifest, job, "refinement-raw")
        by_prompt = {}
        for hypothesis in saved["candidates"]:
            by_prompt.setdefault(hypothesis["prompt_index"], []).append(hypothesis)
        for j, (i, prompt) in enumerate(group):
            old = decode(objects[i]["rle"])
            options = []
            for hypothesis in by_prompt.get(j, []):
                local = decode(hypothesis["rle"])
                full, cut = lift(local, crop, (height, width))
                agreement = iou(full, old)
                ratio = int(full.sum()) / objects[i]["area"]
                positive_x, positive_y = prompt["points"][0]
                if cut or not local.any() or not local[int(positive_y), int(positive_x)]:
                    continue
                if (
                    hypothesis["score"] < settings["quality_min"]
                    or hypothesis["stability"] < settings["stability_min"]
                ):
                    continue
                if (
                    agreement < settings["agreement_min"]
                    or not settings["area_ratio_min"] <= ratio <= settings["area_ratio_max"]
                ):
                    continue
                if (
                    objects[i]["multiscale_support"]
                    and agreement < settings["supported_agreement_min"]
                ):
                    continue
                rank = hypothesis["score"] + 0.25 * agreement - 0.5 * fragment_fraction(full)
                options.append((rank, full, hypothesis, agreement))
            history = dict(
                raw_file=str(path.relative_to(run)),
                points_local=prompt,
                crop=list(crop),
                updated=False,
                initial_rle=objects[i]["rle"],
            )
            if options:
                _, mask, hypothesis, agreement = max(options, key=lambda entry: entry[0])
                objects[i].update(rle=encode(mask), bbox=bbox(mask), area=int(mask.sum()))
                history.update(
                    updated=True,
                    quality=hypothesis["score"],
                    stability=hypothesis["stability"],
                    agreement=agreement,
                )
            else:
                objects[i].setdefault("flags", []).append("no_stable_agreeing_refinement")
            objects[i]["refinement"] = history
            updates.append(
                dict(candidate_id=objects[i]["candidate_id"], updated=history["updated"])
            )
        print(f"Refinement crop {crop}: {len(group)} targets", flush=True)
    chosen = selected["selection"]
    objects, suppressed = dedupe(
        objects,
        chosen["dedupe_iou"],
        chosen.get("dedupe_containment"),
        chosen.get("dedupe_containment_max_area_ratio"),
    )
    annotate_flags(objects, manifest["config"])
    assign_ids(objects)
    result = dict(
        selected,
        masks=objects,
        refinement_settings=settings,
        boundary_updates=updates,
        selection_sha256=file_sha(run / "selected.json"),
        refinement_limit=args.limit,
        refinement_suppressions=suppressed,
        refinement_backend=backend.identity if backend else None,
    )
    write_json(run / "refined.json", result)
    write_json(
        run / "refinement-status.json",
        dict(
            status="complete",
            selection_sha256=result["selection_sha256"],
            result_sha256=file_sha(run / "refined.json"),
        ),
    )
    print(
        f"Retained {sum(u['updated'] for u in updates)} boundary updates. No masks were human-accepted."
    )


def audit(run):
    manifest = manifest_for(run)
    index = read_json(run / "raw" / "index.json")
    points, text = [], []
    for entry in index["records"].values():
        for item in read_prediction(run / entry["file"], expected_sha=entry["sha256"])[
            "candidates"
        ]:
            area = int(decode(item["rle"]).sum())
            destination = text if entry["job"]["kind"] == "text" else points
            destination.append((area, item["score"], item.get("stability")))
    selection = manifest["config"]["selection"]
    eligible = [
        p
        for p in points
        if selection["min_area"] <= p[0]
        and (selection["max_area"] is None or p[0] <= selection["max_area"])
    ]
    sweep = []
    for quality in [None, 0.0, 0.2, 0.5, 0.78, 0.9]:
        for stable in [None, selection["point_stability_min"]]:
            count = sum(
                (quality is None or p[1] >= quality) and (stable is None or p[2] >= stable)
                for p in eligible
            )
            sweep.append(
                dict(quality_min=quality, stability_min=stable, hypotheses_before_dedupe=count)
            )
    result = dict(
        raw_point_hypotheses=len(points),
        raw_text_hypotheses=len(text),
        area_eligible_point_hypotheses=len(eligible),
        point_quality_quantiles=np.quantile(
            [p[1] for p in eligible], [0, 0.1, 0.5, 0.9, 1]
        ).tolist()
        if eligible
        else [],
        sweep=sweep,
        warning="Hypothesis counts before deduplication; not object counts or accuracy. Scores are path/model-specific.",
    )
    write_json(run / "threshold-audit.json", result)
    print(json.dumps(result, indent=2))


def review_sheets(run, size=160, overlap=20, zoom=3):
    manifest = manifest_for(run)
    result = current(run)
    original = Image.open(run / "original.png").convert("RGB")
    array = np.array(original).copy()
    edges = np.zeros((original.height, original.width), bool)
    for obj in result["masks"]:
        mask = decode(obj["rle"])
        edges |= mask & ~ndi.binary_erosion(mask)
    array[edges] = [25, 80, 230]
    overlay = Image.fromarray(array)
    sheets = []
    archive_downstream(run, ["review"])
    for number, (x, y, w, h) in enumerate(tiles(original.width, original.height, size, overlap)):
        box = (x, y, x + w, y + h)
        source, annotated = original.crop(box), overlay.crop(box)
        panel = Image.new("RGB", (w * zoom * 2, h * zoom + 42), "white")
        panel.paste(source.resize((w * zoom, h * zoom), Image.Resampling.NEAREST), (0, 42))
        panel.paste(
            annotated.resize((w * zoom, h * zoom), Image.Resampling.NEAREST), (w * zoom, 42)
        )
        draw = ImageDraw.Draw(panel)
        draw.text(
            (8, 5), f"ORIGINAL | native origin ({x},{y}) | {w}x{h} | zoom {zoom}", fill="black"
        )
        draw.text((w * zoom + 8, 5), "OVERLAY | SAME native crop", fill="black")
        draw.text(
            (8, 23), "Review coordinates must refer to full-image native pixels", fill="black"
        )
        name = f"sheet-{number:03d}.png"
        path = run / "review" / name
        path.parent.mkdir(exist_ok=True)
        panel.save(path)
        sheets.append(
            dict(
                file=name,
                native_crop=[x, y, w, h],
                rendered_zoom=zoom,
                header_height=42,
                source_rendered_origin=[0, 42],
                overlay_rendered_origin=[w * zoom, 42],
            )
        )
    template = dict(
        schema="detailed-segmentation-review/1",
        image_sha256=manifest["canonical_sha256"],
        coordinate_space="native",
        reviewer="fill in reviewer and version",
        observations=[],
    )
    write_json(run / "review" / "template.json", template)
    write_json(
        run / "review" / "sheets.json",
        dict(
            image_sha256=manifest["canonical_sha256"], result_sha256=digest(result), sheets=sheets
        ),
    )
    print(f"Wrote {len(sheets)} coordinate-labelled review sheets to {run / 'review'}")


def validate_result(run):
    manifest = manifest_for(run)
    result = current(run)
    seen = set()
    for obj in result["masks"]:
        mask = decode(obj["rle"])
        if not isinstance(obj.get("id"), str) or re.fullmatch(r"mask-\d{5,}", obj["id"]) is None:
            raise ValueError("Invalid export instance ID")
        if obj["id"] in seen:
            raise ValueError("Duplicate mask ID")
        seen.add(obj["id"])
        if mask.shape != (manifest["height"], manifest["width"]) or not mask.any():
            raise ValueError("Empty or incorrectly sized mask")
        if obj["bbox"] != bbox(mask) or obj["area"] != int(mask.sum()):
            raise ValueError("Stale mask area or pixel-tight bounding box")
        if obj["status"] != "proposal":
            raise ValueError("This workflow must not automatically accept masks")
        if not np.array_equal(decode(encode(mask)), mask):
            raise ValueError("RLE round-trip mismatch")
    return dict(
        masks=len(seen),
        native_size=[manifest["width"], manifest["height"]],
        checks=[
            "canonical image checksum",
            "native dimensions",
            "nonempty masks",
            "unique IDs",
            "pixel-tight boxes including last row/column",
            "mask area",
            "RLE roundtrip",
            "all proposals",
        ],
    )


def export(run, omit_image=False):
    validation = validate_result(run)
    manifest, result = manifest_for(run), current(run)
    out = run / "export"
    out.mkdir(exist_ok=True)
    if not omit_image:
        shutil.copy2(run / "original.png", out / "original.png")
    else:
        (out / "original.png").unlink(missing_ok=True)
    labels = sorted({obj["label"] for obj in result["masks"]})
    categories = {label: i + 1 for i, label in enumerate(labels)}
    annotations, viewer_objects = [], []
    with zipfile.ZipFile(out / "masks.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i, obj in enumerate(result["masks"], 1):
            mask = decode(obj["rle"])
            buffer = io.BytesIO()
            Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
            png = buffer.getvalue()
            if not np.array_equal(np.array(Image.open(io.BytesIO(png))) > 127, mask):
                raise ValueError("PNG round-trip mismatch")
            archive.writestr(f"masks/{obj['id']}.png", png)
            annotations.append(
                dict(
                    id=i,
                    image_id=1,
                    instance_id=obj["id"],
                    category_id=categories[obj["label"]],
                    segmentation=obj["rle"],
                    area=obj["area"],
                    bbox=obj["bbox"],
                    iscrowd=0,
                    status="proposal",
                    review_flags=obj.get("flags", []),
                )
            )
            viewer_objects.append(
                dict(
                    id=obj["id"],
                    label=obj["label"],
                    area=obj["area"],
                    bbox=obj["bbox"],
                    rows=mask_rows(mask),
                    flags=obj.get("flags", []),
                    source=obj["source"],
                    refinement_updated=obj.get("refinement", {}).get("updated", False),
                )
            )
        archive.writestr(
            "README.txt",
            "Native full-resolution binary masks: white belongs to this instance. All are unaccepted proposals. Nested masks overlap intentionally. Counts are not verified object counts.",
        )
    coco = dict(
        info=dict(
            description="Unaccepted detailed segmentation proposals",
            image_sha256=manifest["canonical_sha256"],
        ),
        images=[
            dict(id=1, file_name="original.png", width=manifest["width"], height=manifest["height"])
        ],
        categories=[dict(id=value, name=key) for key, value in categories.items()],
        annotations=annotations,
    )
    write_json(out / "coco.json", coco)
    write_json(out / "proposals.json", result)
    write_json(out / "manifest.json", manifest)
    write_json(out / "validation.json", validation)
    counts = dict(Counter(obj["label"] for obj in result["masks"]))
    write_json(
        out / "data.json",
        dict(
            width=manifest["width"],
            height=manifest["height"],
            counts=counts,
            image_available=not omit_image,
            status="REVIEW DRAFT — NOT GROUND TRUTH",
            synthetic=manifest.get("synthetic_demo", False),
            objects=viewer_objects,
        ),
    )
    shutil.copy2(HERE / "viewer.html", out / "index.html")
    (out / "README.md").write_text(
        "# Detailed segmentation review draft\n\n"
        f"Proposal counts: {counts}. These are NOT verified object counts. No mask is accepted.\n\n"
        "Serve this folder locally to open index.html. PNG masks and COCO RLE are pixel-exact. "
        "Independent human review is still needed for labels, object identity, completeness and boundaries. "
        "Model scores, stability and duplicate suppression do not establish accuracy.\n\n"
        "The private run retains raw candidates, score sweeps and refinement attempts separately. "
        "Preserve those records when reproducing or auditing results. "
        "Check image, annotation and model terms before public distribution.\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(annotations)} verified-format proposals to {out}. Semantic accuracy has not been measured."
    )
    return out


def demo(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Demo output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 240), (42, 45, 40))
    draw = ImageDraw.Draw(image)
    masks = []
    rng = random.Random(42)
    for row in range(7):
        for col in range(11):
            x, y = 14 + col * 26 + (row % 2) * 4, 17 + row * 29
            box = (x, y, x + 20, y + 23)
            draw.ellipse(
                box, fill=(235 + rng.randrange(20), 165 + rng.randrange(55), 40 + rng.randrange(60))
            )
            m = Image.new("L", image.size)
            ImageDraw.Draw(m).ellipse(box, fill=255)
            masks.append(
                record(
                    np.array(m) > 0,
                    "synthetic kernel",
                    dict(kind="synthetic fixture", pass_name="demo", crop=[0, 0, 320, 240]),
                    candidate_id=f"demo-{row}-{col}",
                    flags=[],
                )
            )
    image.save(out / "original.png")
    manifest = dict(
        schema=SCHEMA,
        algorithm=ALGORITHM,
        created_at=now(),
        status="in_progress",
        synthetic_demo=True,
        canonical_sha256=file_sha(out / "original.png"),
        width=320,
        height=240,
        orientation="raw",
        config=load_config(),
        warning="Synthetic plumbing demo, not evidence of model quality.",
    )
    write_json(out / "manifest.json", manifest)
    annotate_flags(masks, manifest["config"])
    assign_ids(masks)
    write_json(
        out / "selected.json",
        dict(
            schema=SCHEMA,
            canonical_sha256=manifest["canonical_sha256"],
            masks=masks,
            status="in_progress",
        ),
    )
    review_sheets(out)
    export(out)


def doctor():
    nvidia = shutil.which("nvidia-smi")
    report = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        uv=shutil.which("uv"),
        git=shutil.which("git"),
        free_disk_gib=round(shutil.disk_usage(Path.cwd()).free / 1024**3, 1),
        packages={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pillow", "pycocotools"]
        },
        torch_available=importlib.util.find_spec("torch") is not None,
        note="This command does not download weights, read credentials or send images to a provider.",
    )
    if nvidia:
        probe = subprocess.run(
            [nvidia, "--query-gpu=index,name,memory.free,driver_version", "--format=csv"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        report["nvidia"] = probe.stdout.strip() or probe.stderr.strip()
    else:
        report["nvidia"] = "nvidia-smi not found; CPU demo/export still work"
    print(json.dumps(report, indent=2))


def model_options(parser):
    parser.add_argument(
        "--device", default="cuda:0", help="NVIDIA GPU, e.g. cuda:0 (no CPU support)"
    )
    parser.add_argument("--precision", choices=["bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint; its SHA256 is recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only setup inspection; never downloads weights")
    p = sub.add_parser("demo", help="CPU-only synthetic plumbing demo; no model or account")
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("prepare", help="Freeze a canonical image and config in a new run")
    p.add_argument("image", nargs="?", type=Path)
    p.add_argument("--random-from", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--orientation", choices=["raw", "exif"], default="raw")
    for name in [
        "plan",
        "audit",
        "validate",
        "review-sheets",
        "export",
        "select",
        "infer",
        "refine",
        "validate-review",
    ]:
        p = sub.add_parser(name)
        p.add_argument("run", type=Path)
        if name == "select":
            p.add_argument(
                "--selection",
                type=Path,
                help="JSON selection-only overrides; raw masks are not rerun",
            )
        if name == "export":
            p.add_argument(
                "--omit-image",
                action="store_true",
                help="Exclude photograph; viewer needs original.png added locally",
            )
        if name in ["infer", "refine"]:
            model_options(p)
        if name == "infer":
            p.add_argument("--review", type=Path)
            p.add_argument("--confirm-review-points", action="store_true")
        if name == "refine":
            p.add_argument("--limit", type=int, help="Explicit cap on attempted instance updates")
        if name == "validate-review":
            p.add_argument("review", type=Path)
    args = parser.parse_args()
    guarded_dispatch(args)


def guarded_dispatch(args):
    if args.command in {
        "prepare",
        "demo",
        "infer",
        "select",
        "refine",
        "audit",
        "review-sheets",
        "export",
    }:
        path = Path(getattr(args, "run", None) or args.out).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.parent / ("." + path.name + ".segmentation.lock")
        try:
            with FileLock(lock_path, timeout=0):
                dispatch(args)
        except Timeout as error:
            raise RuntimeError(
                "Another writer is using this run. Wait or use a different output directory."
            ) from error
    else:
        dispatch(args)


def dispatch(args):
    if args.command == "doctor":
        doctor()
    elif args.command == "demo":
        demo(args.out)
    elif args.command == "prepare":
        selection = None
        if bool(args.image) == bool(args.random_from):
            raise ValueError("Supply exactly one image path or --random-from directory")
        image = args.image
        if args.random_from:
            paths = sorted(
                p
                for p in args.random_from.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
            )
            if not paths:
                raise ValueError("No supported images found")
            image = random.Random(args.seed).choice(paths)
            selection = dict(
                seed=args.seed,
                population=len(paths),
                sorted_population_digest=digest(
                    [str(p.relative_to(args.random_from)) for p in paths]
                ),
                chosen=str(image.relative_to(args.random_from)),
            )
        prepare(image, args.out, load_config(args.config), args.orientation, selection)
    elif args.command == "plan":
        manifest = manifest_for(args.run)
        print(json.dumps(budget_check(planned_jobs(manifest), manifest["config"]), indent=2))
    elif args.command == "infer":
        infer(args.run, args)
    elif args.command == "select":
        select(args.run, args.selection)
    elif args.command == "refine":
        if args.limit is not None and args.limit < 0:
            raise ValueError("Refinement limit cannot be negative")
        refine(args.run, args)
    elif args.command == "audit":
        audit(args.run)
    elif args.command == "review-sheets":
        review_sheets(args.run)
    elif args.command == "validate":
        print(json.dumps(validate_result(args.run), indent=2))
    elif args.command == "validate-review":
        validate_review(read_json(args.review), manifest_for(args.run))
        print(
            "Review schema, image identity and coordinates are valid. Semantic correctness is NOT established."
        )
    elif args.command == "export":
        export(args.run, args.omit_image)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
