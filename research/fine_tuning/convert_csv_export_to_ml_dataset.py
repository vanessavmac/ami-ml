#!/usr/bin/env python
# coding: utf-8

"""
Download Atlantic Forestry CSV export crops into ML dataset folder structure.

Writes `manifest.csv`, `antenna_taxon_issues.csv`, and images under
`atlantic_forestry/{gbif_taxon_key}/{occurrence_id}.jpg`.

Manifest columns (kept rows): `occurrence_id`, `determination_id` (traceability),
`determination_name`, `gbif_taxon_key`, `gbif_resolution_source`, `status`, `local_path`,
`best_detection_url`.

Optional flags: `--skip-download`, `--antenna-api-token`, `--antenna-taxa-cache-json`.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd
from atlantic_dataset_utils import (
    ATLANTIC_DATASET_EXCLUDED_SPECIES,
    DEFAULT_ANTENNA_API_BASE_URL,
    PSEUDO_SPECIES_LABELS,
    AntennaTaxonClient,
    load_quebec_category_map,
    resolve_gbif_taxon_keys_from_dataframe,
)

DEFAULT_ATLANTIC_CSV = "~/data/exports/atlantic-forestry-centre_export-104.csv"
DEFAULT_QUEBEC_MAP = "~/data/models/quebec-vermont_moth-category-map_19Jan2023.json"
DEFAULT_FGRAINED_LABELS = "~/data/ami_traps/insect_crops/fgrained_labels.json"
DEFAULT_OUTPUT_DIR = "~/data/fine_tuning_data_atlantic/atlantic_forestry"


def _download_image(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to download {url}: {exc}", flush=True)
        return False


def convert_csv_export_to_ml_dataset(
    atlantic_csv: str,
    output_dir: str,
    quebec_category_map_json: str,
    fgrained_labels_json: str,
    antenna_api_base_url: str = DEFAULT_ANTENNA_API_BASE_URL,
    antenna_api_token: str | None = None,
    antenna_taxa_cache_json: str | None = None,
    skip_download: bool = False,
) -> pd.DataFrame:
    """
    Filter trainable rows, resolve GBIF keys via Antenna API, download crops, write manifest.

    Trainable rows are:
    - Verified
    - Not a pseudo label
    - Not an excluded species
    - Not a non-binomial
    - In the Quebec map
    - Has a best detection URL
    - Has a GBIF taxon key
    - Has a local path
    - Has a status
    - Has a reason
    - Has a local path
    - Has a best detection URL
    """
    quebec_map = load_quebec_category_map(quebec_category_map_json)
    df = pd.read_csv(atlantic_csv)

    output_parent = Path(output_dir).parent
    cache_path = (
        Path(antenna_taxa_cache_json)
        if antenna_taxa_cache_json
        else output_parent / "antenna_taxon_cache.json"
    )

    manifest_rows = []
    trainable_rows = []

    for _, row in df.iterrows():
        reason = None
        if row.get("verification_status") is not True:
            reason = "not_verified"
        elif row["determination_name"] in PSEUDO_SPECIES_LABELS:
            reason = "pseudo_label"
        elif row["determination_name"] in ATLANTIC_DATASET_EXCLUDED_SPECIES:
            reason = "excluded_species"
        elif " " not in str(row["determination_name"]):
            reason = "non_binomial"
        elif row["determination_name"] not in quebec_map:
            reason = "not_in_quebec_map"
        elif pd.isna(row.get("best_detection_url")) or not row.get(
            "best_detection_url"
        ):
            reason = "missing_image_url"

        if reason:
            manifest_rows.append(
                {
                    "occurrence_id": row.get("id"),
                    "determination_id": row.get("determination_id"),
                    "determination_name": row.get("determination_name"),
                    "gbif_taxon_key": None,
                    "status": "skipped",
                    "reason": reason,
                    "local_path": None,
                }
            )
            continue

        trainable_rows.append(row)

    if trainable_rows:
        trainable_df = pd.DataFrame(trainable_rows)
        client = AntennaTaxonClient(
            base_url=antenna_api_base_url,
            api_token=antenna_api_token,
            cache_path=cache_path,
        )
        trainable_df = resolve_gbif_taxon_keys_from_dataframe(
            trainable_df,
            client=client,
            fgrained_labels_path=fgrained_labels_json,
            issues_report_path=output_parent / "antenna_taxon_issues.csv",
        )
    else:
        trainable_df = pd.DataFrame()

    for _, row in trainable_df.iterrows():
        gbif_taxon_key = row["gbif_taxon_key"]
        determination_id = str(int(float(row["determination_id"])))
        occurrence_id = str(int(float(row["id"])))
        local_path = Path(output_dir) / gbif_taxon_key / f"{occurrence_id}.jpg"
        url = row["best_detection_url"]

        downloaded = True
        if not skip_download:
            downloaded = _download_image(url, local_path)

        manifest_rows.append(
            {
                "occurrence_id": occurrence_id,
                "determination_id": determination_id,
                "determination_name": row["determination_name"],
                "gbif_taxon_key": gbif_taxon_key,
                "gbif_resolution_source": row.get("gbif_resolution_source"),
                "status": "kept" if downloaded else "skipped",
                "reason": None if downloaded else "download_failed",
                "local_path": str(local_path) if downloaded else None,
                "best_detection_url": url,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_parent / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    kept = manifest[manifest["status"] == "kept"]
    print(
        f"Manifest: {len(kept)} kept / {len(manifest)} total " f"-> {manifest_path}",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Atlantic Forestry CSV export to ML dataset folders."
    )
    parser.add_argument("--atlantic-csv", default=DEFAULT_ATLANTIC_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quebec-category-map-json", default=DEFAULT_QUEBEC_MAP)
    parser.add_argument("--fgrained-labels-json", default=DEFAULT_FGRAINED_LABELS)
    parser.add_argument(
        "--antenna-api-base-url",
        default=DEFAULT_ANTENNA_API_BASE_URL,
    )
    parser.add_argument("--antenna-api-token", default=None)
    parser.add_argument(
        "--antenna-taxa-cache-json",
        default=None,
        help="Path to antenna_taxon_cache.json (default: beside manifest).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only write manifest without downloading images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_csv_export_to_ml_dataset(
        atlantic_csv=args.atlantic_csv,
        output_dir=args.output_dir,
        quebec_category_map_json=args.quebec_category_map_json,
        fgrained_labels_json=args.fgrained_labels_json,
        antenna_api_base_url=args.antenna_api_base_url,
        antenna_api_token=args.antenna_api_token,
        antenna_taxa_cache_json=args.antenna_taxa_cache_json,
        skip_download=args.skip_download,
    )
