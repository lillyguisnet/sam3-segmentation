"""Pinned SAM 3 still-image adapter; no shared service or remote credentials."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from core import encode, file_sha, finite, stability

SAM3_SOURCE = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
WEIGHTS_REPO = "facebook/sam3"
WEIGHTS_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def prompt_batches(prompts, width, height, batch_size):
    """Group by true prompt arity: repeated not-a-point tokens can alter masks."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    by_arity = {}
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict) or set(prompt) != {"points", "labels"}:
            raise ValueError("Each object prompt needs points and labels")
        points, labels = prompt["points"], prompt["labels"]
        if (
            not isinstance(points, list)
            or not isinstance(labels, list)
            or not points
            or len(points) != len(labels)
        ):
            raise ValueError("Prompt coordinates and labels must be nonempty equal-length lists")
        if not all(type(label) is int and label in (0, 1) for label in labels) or 1 not in labels:
            raise ValueError("Prompts need at least one positive and only 1/0 labels")
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(finite(value) for value in point)
            ):
                raise ValueError("Point prompts must contain finite [x,y] pairs")
            if not 0 <= point[0] < width or not 0 <= point[1] < height:
                raise ValueError("Point prompt outside the crop")
        by_arity.setdefault(len(points), []).append(i)
    batches = []
    for indices in by_arity.values():
        for offset in range(0, len(indices), batch_size):
            group = indices[offset : offset + batch_size]
            batches.append(
                (
                    group,
                    np.array([prompts[i]["points"] for i in group], np.float32),
                    np.array([prompts[i]["labels"] for i in group], np.int32),
                )
            )
    return batches


class Sam3Backend:
    def __init__(self, device="cuda:0", checkpoint=None, precision="bf16", batch_size=24, seed=0):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable. Run doctor in the model environment; check the NVIDIA driver."
                )
            torch.cuda.set_device(device)
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "This GPU does not support BF16; use --precision fp32 and a smaller batch."
                )
        elif device != "cpu":
            raise ValueError("Validated device paths are cpu and cuda:N; MPS is not implemented")
        elif precision != "fp32":
            raise ValueError(
                "CPU mode requires --precision fp32 (very slow; no throughput promise)"
            )
        torch.set_num_threads(8)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        downloaded_reference = checkpoint is None
        if checkpoint is None:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                WEIGHTS_REPO, filename="sam3.pt", revision=WEIGHTS_REVISION
            )
        distribution = importlib.metadata.distribution("sam3")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
        if installed_commit != SAM3_SOURCE:
            raise RuntimeError(
                "Installed SAM 3 source does not attest the pinned commit; use the supplied locked environment"
            )
        import sam3

        source_root = Path(sam3.__file__).resolve().parent
        tree_hash = hashlib.sha256()
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".gz", ".json", ".yaml", ".yml"}:
                tree_hash.update(str(path.relative_to(source_root)).encode())
                tree_hash.update(file_sha(path).encode())
        hardware = dict(
            machine=platform.machine(),
            python=platform.python_version(),
            processor=platform.processor(),
            cudnn=torch.backends.cudnn.version(),
        )
        if device.startswith("cuda"):
            properties = torch.cuda.get_device_properties(device)
            hardware.update(
                gpu_name=properties.name,
                capability=[properties.major, properties.minor],
                total_memory=properties.total_memory,
                uuid=str(getattr(properties, "uuid", "unavailable")),
            )
            executable = shutil.which("nvidia-smi")
            if executable:
                probe = subprocess.run(
                    [executable, "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=15,
                )
                hardware["driver_inventory"] = (
                    probe.stdout.strip() if probe.returncode == 0 else "query unavailable"
                )
        self.identity = {
            "backend": "sam3-image",
            "source_commit": installed_commit,
            "installed_source_tree_sha256": tree_hash.hexdigest(),
            "hardware": hardware,
            "environment": {
                key: os.environ.get(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "NVIDIA_VISIBLE_DEVICES",
                    "CUBLAS_WORKSPACE_CONFIG",
                ]
            },
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "weights_origin": "pinned-hub" if downloaded_reference else "user-local-file",
            "weights_repo": WEIGHTS_REPO if downloaded_reference else None,
            "reference_weights_revision": WEIGHTS_REVISION,
            "checkpoint_sha256": file_sha(checkpoint),
            "device": device,
            "precision": precision,
            "batch_size": batch_size,
            "seed": seed,
            "tf32": True,
            "raw_text_api_threshold": 0.0,
            "hole_filling": 0,
            "speck_removal": 0,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "torchvision", "numpy", "scipy", "pillow", "pycocotools"]
            },
            "torch_cuda": torch.version.cuda,
        }
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_inst_interactivity=True,
            device="cpu",
            compile=False,
        ).to(device)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=0.0)
        predictor = self.model.inst_interactive_predictor
        predictor._transforms.max_hole_area = 0
        predictor._transforms.max_sprinkle_area = 0
        self._image_key = None
        self.state = None

    def context(self):
        if self.device.startswith("cuda") and self.precision == "bf16":
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return contextlib.nullcontext()

    def image_state(self, image):
        key = (image.size, hashlib.sha256(image.tobytes()).hexdigest())
        if key != self._image_key:
            self.state = self.processor.set_image(image)
            self._image_key = key
        return self.state

    def text(self, image, phrase):
        with self.torch.inference_mode(), self.context():
            state = self.image_state(image)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(phrase, state)
            masks = state["masks"].squeeze(1).cpu().numpy()
            scores = state["scores"].float().cpu().numpy()
            boxes = state["boxes"].float().cpu().numpy()
        return [
            dict(
                rle=encode(mask),
                score=float(score),
                stability=None,
                model_box_xyxy=box.tolist(),
                hypothesis=i,
            )
            for i, (mask, score, box) in enumerate(zip(masks, scores, boxes, strict=True))
        ]

    def points(self, image, prompts):
        """Each list entry is an independent object prompt, never one joint object."""
        result = []
        batches = prompt_batches(prompts, image.width, image.height, self.batch_size)
        if not batches:
            return result
        with self.torch.inference_mode(), self.context():
            state = self.image_state(image)
            for indices, coords, labels in batches:
                logits, scores, _ = self.model.predict_inst(
                    state,
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=True,
                    return_logits=True,
                )
                if logits.ndim == 3:
                    logits, scores = logits[None], scores[None]
                for j, (masks, qualities) in enumerate(zip(logits, scores, strict=True)):
                    for hypothesis, (mask_logits, quality) in enumerate(
                        zip(masks, qualities, strict=True)
                    ):
                        result.append(
                            dict(
                                rle=encode(mask_logits > 0),
                                score=float(quality),
                                stability=stability(mask_logits),
                                prompt_index=indices[j],
                                hypothesis=hypothesis,
                            )
                        )
        return sorted(
            result, key=lambda candidate: (candidate["prompt_index"], candidate["hypothesis"])
        )
