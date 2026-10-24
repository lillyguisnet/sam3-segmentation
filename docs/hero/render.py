"""Draw the README picture: the photo beside its kernel proposals, one colour each."""

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from scipy import ndimage

export, output = Path(sys.argv[1]), Path(sys.argv[2])
coco = json.loads((export / "coco.json").read_text())
image = np.asarray(Image.open(export / "original.png").convert("RGB")).astype(np.float32)
overlay = image.copy()
edges = np.zeros(image.shape[:2], bool)
kernels = [a for a in coco["annotations"] if a["category_id"] == 2]
rng = np.random.default_rng(3)
for annotation in kernels:
    mask = mask_utils.decode(annotation["segmentation"]).astype(bool)
    colour = np.array(colorsys.hsv_to_rgb(rng.random(), 0.75, 1.0)) * 255
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * colour
    edges |= mask & ~ndimage.binary_erosion(mask)
overlay[edges] *= 0.3

x0, y0, x1, y1 = 150, 40, 870, 520
left = Image.fromarray(image[y0:y1, x0:x1].astype(np.uint8))
right = Image.fromarray(overlay[y0:y1, x0:x1].clip(0, 255).astype(np.uint8))
gap, (width, height) = 8, left.size
picture = Image.new("RGB", (2 * width + gap, height), "white")
picture.paste(left, (0, 0))
picture.paste(right, (width + gap, 0))
draw = ImageDraw.Draw(picture)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except OSError:
    font = ImageFont.load_default(22)
for x, text in [
    (0, "Photo (CC0)"),
    (width + gap, f"{len(kernels):,} kernel proposals, unreviewed"),
]:
    text_width = draw.textlength(text, font=font)
    draw.rounded_rectangle((x + 12, 12, x + 32 + text_width, 50), 8, fill=(0, 0, 0))
    draw.text((x + 22, 17), text, font=font, fill="white")
picture.save(output, quality=86, optimize=True)
print(f"{len(kernels)} kernels -> {output}")
