import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_SOURCES = [
    "data/train/flickr2k/Flickr2K",
    "data/train/div2k/DIV2K_train_HR",
    "data/train/clic2020train",
    "data/train/clic2020val",
    "data/train/clic2021test",
    # "data/train/lsdir10k",
]
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build an HDF5 training dataset with image metadata for caption-conditioned training.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/train/dataset2_caption.hdf5",
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Training image directories. Defaults match the original build_h5.py layout.",
    )
    return parser.parse_args()


def iter_image_files(source_dir):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[Skip] Missing directory: {source_path}")
        return []

    files = []
    for item in sorted(source_path.iterdir()):
        if item.is_file() and item.suffix.lower() in VALID_SUFFIXES:
            files.append(item)
    return files


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        idx = 0
        for source_dir in args.sources:
            image_files = iter_image_files(source_dir)
            source_path = Path(source_dir)
            for image_path in tqdm(image_files, desc=str(source_path)):
                
                image = Image.open(image_path).convert("RGB")

                sample = h5_file.create_group(str(idx))
                sample.create_dataset("image", data=image, dtype=np.uint8)
                sample.attrs["image_name"] = image_path.name
                sample.attrs["source_dir"] = str(source_path)
                idx += 1

    print(f"[Done] Wrote {idx} images to {output_path}")


if __name__ == "__main__":
    main()
