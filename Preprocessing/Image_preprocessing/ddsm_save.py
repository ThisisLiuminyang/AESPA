import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_image(array):
    image = np.squeeze(array).astype(np.float32)
    image -= image.min()
    maximum = image.max()
    if maximum > 0:
        image /= maximum
    return np.rint(image * 255).astype(np.uint8)


def convert(manifest, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open(encoding="utf-8") as file:
        records = [line.strip() for line in file if line.strip()]
    converted = 0
    for record in records:
        _, source_path = record.split(maxsplit=1)
        image = sitk.ReadImage(source_path)
        array = sitk.GetArrayFromImage(image)
        normalized = normalize_image(array)
        Image.fromarray(normalized).save(
            output_dir / f"image_{converted:08d}.png"
        )
        converted += 1
    return converted


def main():
    args = parse_args()
    return convert(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
