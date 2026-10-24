#!/usr/bin/env python3
"""Explicit opt-in real-model API check on synthetic shapes, not an accuracy test."""

import sys

# Do not write __pycache__ into the (possibly shared or read-only) skill folder.
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
from core import decode, encode, iou, write_json
from PIL import Image, ImageDraw
from sam3_backend import Sam3Backend
from segment import model_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_options(parser)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new empty output directory")
    image = Image.new("RGB", (128, 128), (35, 40, 35))
    draw = ImageDraw.Draw(image)
    draw.ellipse((15, 20, 52, 64), fill=(245, 205, 60))
    draw.ellipse((70, 55, 110, 101), fill=(245, 205, 60))
    prompts = [
        dict(points=[[33, 40]], labels=[1]),
        dict(points=[[33, 40], [89, 77]], labels=[1, 0]),
        dict(points=[[89, 77], [33, 40], [63, 18], [10, 110]], labels=[1, 0, 0, 0]),
        dict(points=[[89, 77]], labels=[1]),
    ]
    backend = Sam3Backend(
        device=args.device,
        checkpoint=args.checkpoint,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    text = backend.text(image, "yellow oval")
    batched = backend.points(image, prompts)
    singles = [backend.points(image, [prompt]) for prompt in prompts]
    if len(batched) != 3 * len(prompts) or any(len(group) != 3 for group in singles):
        raise RuntimeError(
            "The adapter did not return three hypotheses per independent point prompt"
        )
    for candidate in text + batched + [item for group in singles for item in group]:
        mask = decode(candidate["rle"])
        if mask.shape != (128, 128) or not np.isfinite(candidate["score"]):
            raise RuntimeError("Invalid native mask geometry or non-finite model signal")
        if not np.array_equal(decode(encode(mask)), mask):
            raise RuntimeError("Mask codec failed its pixel round trip")
    paired = []
    for i, group in enumerate(singles):
        batch_group = [c for c in batched if c["prompt_index"] == i]
        for first, second in zip(batch_group, group, strict=True):
            a, b = decode(first["rle"]), decode(second["rle"])
            paired.append(1.0 if not a.any() and not b.any() else iou(a, b))
    summary = dict(
        purpose="Real API/geometry and batching smoke check on synthetic shapes; NOT measured segmentation accuracy",
        model=backend.identity,
        text_hypotheses=len(text),
        independent_prompts=len(prompts),
        batched_hypotheses=len(batched),
        singleton_hypotheses=sum(map(len, singles)),
        batch_single_ious=paired,
        median_batch_single_iou=float(np.median(paired)),
        minimum_batch_single_iou=float(min(paired)),
        warning="Inspect every hypothesis, not only the median. Reduce to --batch-size 1 if individual masks disagree substantially.",
        checks=[
            "real text API",
            "independent prompts",
            "mixed positive/negative prompt lengths",
            "native geometry",
            "finite signals",
            "COCO RLE pixel round trips",
        ],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    image.save(args.out / "synthetic-input.png")
    write_json(args.out / "results.json", summary)
    write_json(
        args.out / "hypotheses.json.gz",
        dict(text=text, batched=batched, singleton=singles, prompts=prompts),
    )
    print(json.dumps(summary, indent=2))
    if summary["median_batch_single_iou"] < 0.95 or summary["minimum_batch_single_iou"] < 0.85:
        raise SystemExit(
            "Large batch/single disagreement: inspect results before trusting batching on this setup"
        )


if __name__ == "__main__":
    main()
