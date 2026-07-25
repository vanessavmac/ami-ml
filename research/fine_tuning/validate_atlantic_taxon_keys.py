#!/usr/bin/env python
# coding: utf-8

"""
Validate Atlantic Forestry GBIF taxon keys before webdataset conversion and training.

Checks: non-null `gbif_taxon_key` on kept manifest rows; every dataset split `taxonkey` in
bridge JSON (created by `build_atlantic_category_map.py`); on-disk images at
`{images_subdir}/{taxonkey}/{filename}`; overlap species resolved key == fgrained `acceptedTaxonKey` from AMI-Traps.
Warns if `antenna_taxon_issues.csv`lists taxa needing platform fixes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from atlantic_dataset_utils import (
    DEFAULT_ANTENNA_API_BASE_URL,
    ISSUE_NEEDS_ANTENNA_FIX,
    AntennaTaxonClient,
    build_fgrained_species_to_accepted_key,
    load_fgrained_labels,
    load_quebec_category_map,
    normalize_taxon_key,
    resolve_gbif_taxon_key_for_determination,
)

DEFAULT_DATA_DIR = "~/vanessa/data/fine_tuning_data_atlantic"
DEFAULT_MANIFEST = f"{DEFAULT_DATA_DIR}/manifest.csv"
DEFAULT_BRIDGE = f"{DEFAULT_DATA_DIR}/taxon_to_training_idx.json"
DEFAULT_FGRAINED_LABELS = "~/vanessa/data/ami_traps/insect_crops/fgrained_labels.json"
DEFAULT_QUEBEC_MAP = "~/vanessa/data/models/quebec-vermont_moth-category-map_19Jan2023.json"
DEFAULT_AMI_TRAPS_TEST_CSV = "~/vanessa/data/fine_tuning_data/test.csv"
DEFAULT_IMAGES_SUBDIR = "atlantic_forestry"


def _fail(errors: list[str]) -> None:
    print("Atlantic GBIF taxon key validation FAILED:", flush=True)
    for error in errors:
        print(f"  - {error}", flush=True)
    sys.exit(1)


def _build_ami_traps_species_to_key(
    test_csv_path: str, fgrained_labels_json: str
) -> dict[str, str]:
    fgrained_ref = build_fgrained_species_to_accepted_key(
        load_fgrained_labels(fgrained_labels_json)
    )
    test_df = pd.read_csv(test_csv_path)
    test_df["taxonkey"] = test_df["taxonkey"].apply(normalize_taxon_key)
    test_keys = set(test_df["taxonkey"].dropna())
    return {species: key for species, key in fgrained_ref.items() if key in test_keys}


def _warn_antenna_taxon_issues(data_dir: Path) -> None:
    issues_path = data_dir / "antenna_taxon_issues.csv"
    if not issues_path.exists():
        return
    issues_df = pd.read_csv(issues_path)
    if "issues" not in issues_df.columns:
        return
    needs_fix = issues_df[
        issues_df["issues"].astype(str).str.contains(ISSUE_NEEDS_ANTENNA_FIX, na=False)
    ]
    if len(needs_fix):
        print(
            f"\nNote: {len(needs_fix)} Antenna taxa in {issues_path} need platform fixes "
            f"(resolved via fallbacks). See recommended_action column.",
            flush=True,
        )


def validate_atlantic_taxon_keys(
    data_dir: str,
    manifest_csv: str,
    bridge_json: str,
    fgrained_labels_json: str,
    quebec_category_map_json: str,
    ami_traps_test_csv: str,
    images_subdir: str = DEFAULT_IMAGES_SUBDIR,
    antenna_api_base_url: str = DEFAULT_ANTENNA_API_BASE_URL,
    antenna_api_token: str | None = None,
    antenna_taxa_cache_json: str | None = None,
) -> bool:
    errors: list[str] = []
    data_path = Path(data_dir)
    manifest_path = Path(manifest_csv)

    if not manifest_path.exists():
        errors.append(f"manifest not found: {manifest_path}")
        _fail(errors)

    manifest = pd.read_csv(manifest_path)
    kept = manifest[manifest["status"] == "kept"]

    if "gbif_taxon_key" not in manifest.columns:
        errors.append("manifest.csv missing gbif_taxon_key column")
    else:
        missing = kept[kept["gbif_taxon_key"].isna()]
        if len(missing):
            errors.append(f"{len(missing)} kept manifest rows have null gbif_taxon_key")

    if not Path(bridge_json).exists():
        errors.append(f"bridge file not found: {bridge_json}")
        bridge: dict[str, int] = {}
    else:
        with open(bridge_json, "r", encoding="utf-8") as f:
            bridge = json.load(f)

    quebec_map = load_quebec_category_map(quebec_category_map_json)
    quebec_indices = set(quebec_map.values())
    for taxon_key, class_idx in bridge.items():
        if class_idx not in quebec_indices:
            errors.append(
                f"bridge key {taxon_key} maps to invalid Quebec class index {class_idx}"
            )

    images_root = data_path / images_subdir
    for split in ("train", "val"):
        split_path = data_path / f"{split}.csv"
        if not split_path.exists():
            errors.append(f"missing split file: {split_path}")
            continue

        split_df = pd.read_csv(split_path)
        for _, row in split_df.iterrows():
            taxon_key = normalize_taxon_key(row["taxonkey"])
            if taxon_key is None:
                errors.append(f"{split}.csv row has null taxonkey: {row['filename']}")
                continue
            if taxon_key not in bridge:
                errors.append(
                    f"{split}.csv taxonkey {taxon_key} missing from taxon_to_training_idx.json"
                )
            image_path = images_root / taxon_key / row["filename"]
            if not image_path.is_file():
                errors.append(f"missing image: {image_path}")

    if not kept.empty and "gbif_taxon_key" in kept.columns:
        fgrained_ref = build_fgrained_species_to_accepted_key(
            load_fgrained_labels(fgrained_labels_json)
        )
        cache_path = (
            Path(antenna_taxa_cache_json)
            if antenna_taxa_cache_json
            else manifest_path.parent / "antenna_taxon_cache.json"
        )
        client = AntennaTaxonClient(
            base_url=antenna_api_base_url,
            api_token=antenna_api_token,
            cache_path=cache_path,
        )

        manifest_by_species = (
            kept.groupby("determination_name")
            .agg(
                gbif_taxon_key=("gbif_taxon_key", "first"),
                determination_id=("determination_id", "first"),
                occurrence_id=("occurrence_id", "first"),
            )
            .reset_index()
        )

        for _, row in manifest_by_species.iterrows():
            species = row["determination_name"]
            manifest_key = str(int(float(row["gbif_taxon_key"])))
            try:
                result = resolve_gbif_taxon_key_for_determination(
                    client,
                    row["determination_id"],
                    species_name=species,
                    sample_occurrence_id=row["occurrence_id"],
                    fgrained_ref=fgrained_ref,
                )
                resolved = result.gbif_taxon_key
            except Exception as exc:  # noqa: BLE001
                errors.append(f"GBIF key resolution failed for {species!r}: {exc}")
                continue
            if manifest_key != resolved:
                errors.append(
                    f"manifest gbif_taxon_key for {species!r} is {manifest_key}, "
                    f"expected {resolved} from Antenna API"
                )
            if species in fgrained_ref and manifest_key != fgrained_ref[species]:
                errors.append(
                    f"overlap species {species!r}: manifest={manifest_key}, "
                    f"fgrained={fgrained_ref[species]}"
                )

        ami_traps_species_keys = _build_ami_traps_species_to_key(
            ami_traps_test_csv, fgrained_labels_json
        )
        overlap_species = set(manifest_by_species["determination_name"]) & set(
            ami_traps_species_keys
        )
        for species in sorted(overlap_species):
            manifest_key = str(
                int(
                    float(
                        manifest_by_species.loc[
                            manifest_by_species["determination_name"] == species,
                            "gbif_taxon_key",
                        ].iloc[0]
                    )
                )
            )
            ami_key = ami_traps_species_keys[species]
            if manifest_key != ami_key:
                errors.append(
                    f"overlap species {species!r}: atlantic={manifest_key}, "
                    f"ami_traps_test={ami_key}"
                )

    if errors:
        _fail(errors)

    _warn_antenna_taxon_issues(data_path)
    print("Atlantic GBIF taxon key validation passed.", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Atlantic Forestry GBIF taxon keys and dataset consistency."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest-csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--bridge-json", default=DEFAULT_BRIDGE)
    parser.add_argument("--fgrained-labels-json", default=DEFAULT_FGRAINED_LABELS)
    parser.add_argument("--quebec-category-map-json", default=DEFAULT_QUEBEC_MAP)
    parser.add_argument("--ami-traps-test-csv", default=DEFAULT_AMI_TRAPS_TEST_CSV)
    parser.add_argument("--images-subdir", default=DEFAULT_IMAGES_SUBDIR)
    parser.add_argument("--antenna-api-base-url", default=DEFAULT_ANTENNA_API_BASE_URL)
    parser.add_argument("--antenna-api-token", default=None)
    parser.add_argument("--antenna-taxa-cache-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_atlantic_taxon_keys(
        data_dir=args.data_dir,
        manifest_csv=args.manifest_csv,
        bridge_json=args.bridge_json,
        fgrained_labels_json=args.fgrained_labels_json,
        quebec_category_map_json=args.quebec_category_map_json,
        ami_traps_test_csv=args.ami_traps_test_csv,
        images_subdir=args.images_subdir,
        antenna_api_base_url=args.antenna_api_base_url,
        antenna_api_token=args.antenna_api_token,
        antenna_taxa_cache_json=args.antenna_taxa_cache_json,
    )
