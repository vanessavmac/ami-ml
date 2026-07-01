#!/usr/bin/env python
# coding: utf-8

"""Conversion of ML structured data to the WebDataset format"""

import argparse
import json
import os
from pathlib import Path
from typing import Generator

import dotenv
import pandas as pd
import PIL
import webdataset as wds
from PIL import Image

dotenv.load_dotenv()


def _get_image(image_path: Path) -> Image.Image | None:
    """Read an image from the given path.
    Args:
        image_path (Path): Path to the image file.
    Returns:
        Image.Image: The loaded image.
    """

    image = None

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
    except PIL.UnidentifiedImageError:
        print(f"Unidentified Image Error on file {image_path}", flush=True)
    except OSError:
        print(f"OSError Error on file {image_path}", flush=True)

    return image


def _normalize_taxon_key(value) -> str | None:
    if pd.isna(value):
        return None
    return str(int(float(value)))


def _create_samples(
    dataset_dir: Path,
    category_map: dict,
    split_type: str,
    images_subdir: str,
) -> Generator:
    """Create samples for the webdataset.

    Args:
        dataset_dir (Path): Directory containing the dataset.
        category_map (dict): Mapping of taxon keys to labels.
        split_type (str): Type of split (train, val, test).
        images_subdir (str): Subdirectory containing the images.
    """

    # Read the split files
    dataset_df = pd.read_csv(dataset_dir / (split_type + ".csv"))

    for _, row in dataset_df.iterrows():
        taxon_key = _normalize_taxon_key(row["taxonkey"])
        if taxon_key is None:
            continue

        filename = row["filename"]
        image = _get_image(dataset_dir / images_subdir / taxon_key / filename)
        label = category_map.get(taxon_key, None)
        if not label:
            print(f"Label not found for taxon key {taxon_key}", flush=True)
        if image and label:
            sample = {"__key__": Path(filename).stem, "jpg": image, "cls": label}
            yield sample


def _write_samples_to_sink(
    samples,
    webdataset_dir: Path,
    split_type: str,
    max_shard_size: int = 25 * 1024 * 1024,
) -> None:
    """Write samples to the sink.

    Args:
        samples (Generator): Generator object containing the files to be written.
        webdataset_dir (Path): Directory to save the webdataset.
        split_type (str): Type of split (train, val, test).
        max_shard_size (int): Maximum size of each shard in bytes.

    Returns:
        None: The function saves the samples in the specified directory.
    """
    webdataset_pattern = str(webdataset_dir / f"{split_type}-%06d.tar")
    webdataset_dir.mkdir(parents=True, exist_ok=True)

    with wds.ShardWriter(webdataset_pattern, maxsize=max_shard_size) as sink:
        for sample in samples:
            sink.write(sample)


def convert_to_webdataset(
    fine_tuning_data_dir: str,
    category_map_f: str,
    images_subdir: str = "ami_traps",
) -> None:
    """Main function to convert fine-tuning camera trap data to webdataset format.

    Args:
        fine_tuning_data_dir (str): Directory containing split CSVs.
        category_map_f (str): JSON mapping taxon key (str) to class index.
        images_subdir (str): Subdirectory under fine_tuning_data_dir with taxon folders.

    Returns:
        None: The function processes and saves the webdataset in the specified directory.
    """

    # Read the category map
    with open(category_map_f, "r", encoding="utf-8") as f:
        category_map = json.load(f)

    # Create samples for the webdataset
    fine_tuning_data_path = Path(fine_tuning_data_dir)
    for split_type in ("train", "val", "test"):
        split_csv = fine_tuning_data_path / f"{split_type}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")

        samples = _create_samples(
            fine_tuning_data_path,
            category_map,
            split_type,
            images_subdir,
        )
        _write_samples_to_sink(
            samples,
            fine_tuning_data_path / "webdataset" / split_type,
            split_type,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ML dataset to WebDataset.")
    parser.add_argument(
        "--fine-tuning-data-dir",
        default="~/data/fine_tuning_data",
    )
    parser.add_argument(
        "--category-map-f",
        default="~/data/fine_tuning_data/taxon_to_quebec_idx.json",
    )
    parser.add_argument(
        "--images-subdir",
        default="ami_traps",
        help="Subdirectory containing taxon-key image folders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_webdataset(
        fine_tuning_data_dir=args.fine_tuning_data_dir,
        category_map_f=args.category_map_f,
        images_subdir=args.images_subdir,
    )
