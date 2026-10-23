from argparse import Namespace

import numpy as np
import pytest
import segment
from core import canonical_image
from filelock import FileLock
from PIL import Image, ImageOps
from sam3_backend import prompt_batches


def test_prompt_batches_do_not_pad_unrelated_arity():
    prompts = [
        dict(points=[[1, 2]], labels=[1]),
        dict(points=[[1, 2], [4, 5]], labels=[1, 0]),
        dict(points=[[7, 8]], labels=[1]),
    ]
    batches = prompt_batches(prompts, 10, 10, 24)
    assert [indices for indices, _, _ in batches] == [[0, 2], [1]]
    assert [coordinates.shape for _, coordinates, _ in batches] == [(2, 1, 2), (1, 2, 2)]
    assert all((labels >= 0).all() for _, _, labels in batches)


@pytest.mark.parametrize(
    "prompt",
    [
        dict(points=[[10, 2]], labels=[1]),
        dict(points=[[1, 2]], labels=[0]),
        dict(points=[[1, 2]], labels=[1, 0]),
    ],
)
def test_invalid_model_prompts_rejected_before_inference(prompt):
    with pytest.raises(ValueError):
        prompt_batches([prompt], 10, 10, 24)


def test_mutating_cli_refuses_another_writer(tmp_path):
    out = tmp_path / "demo"
    lock = tmp_path / ".demo.segmentation.lock"
    with FileLock(lock, timeout=0):
        with pytest.raises(RuntimeError, match="Another writer"):
            segment.guarded_dispatch(Namespace(command="demo", out=out))
    assert not out.exists()


def test_explicit_exif_policy_keeps_models_and_pixels_aligned(tmp_path):
    array = np.arange(7 * 4 * 3, dtype=np.uint8).reshape(7, 4, 3)
    image = Image.fromarray(array)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    raw = canonical_image(path, "raw")
    upright = canonical_image(path, "exif")
    assert raw.size == (4, 7)
    assert upright.size == (7, 4)
    with Image.open(path) as decoded:
        assert np.array_equal(
            np.array(upright), np.array(ImageOps.exif_transpose(decoded).convert("RGB"))
        )
    assert np.array_equal(np.array(raw), array)
