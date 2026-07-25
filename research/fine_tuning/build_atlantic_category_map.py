#!/usr/bin/env python
# coding: utf-8

"""Build GBIF taxon key -> Quebec class index map for Atlantic Forestry fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from atlantic_dataset_utils import (
    DEFAULT_ANTENNA_API_BASE_URL,
    AntennaTaxonClient,
    filter_atlantic_trainable_rows,
    load_quebec_category_map,
    resolve_gbif_taxon_keys_from_dataframe,
)

DEFAULT_ATLANTIC_CSV = "~/vanessa/data/exports/atlantic-forestry-centre_export-104.csv"
DEFAULT_MANIFEST = "~/vanessa/data/fine_tuning_data_atlantic/manifest.csv"
DEFAULT_QUEBEC_MAP = "~/vanessa/data/models/quebec-vermont_moth-category-map_19Jan2023.json"
DEFAULT_FGRAINED_LABELS = "~/vanessa/data/ami_traps/insect_crops/fgrained_labels.json"
DEFAULT_OUTPUT = "~/vanessa/data/fine_tuning_data_atlantic/taxon_to_training_idx.json"


def build_atlantic_category_map(
    quebec_category_map_json: str,
    output_path: str,
    atlantic_csv: str | None = None,
    manifest_csv: str | None = None,
    fgrained_labels_json: str | None = None,
    antenna_api_base_url: str = DEFAULT_ANTENNA_API_BASE_URL,
    antenna_api_token: str | None = None,
    antenna_taxa_cache_json: str | None = None,
) -> dict[str, int]:
    """
    Build gbif_taxon_key (str) -> Quebec class index for trainable rows.
    Trainable rows are retrieved from the manifest.csv or atlantic_csv.

    Args:
    - quebec_category_map_json: path to the Quebec category map JSON file
    - output_path: path to the output JSON file
    - atlantic_csv: path to the Atlantic Forestry CSV file
    - manifest_csv: path to the manifest CSV file
    - fgrained_labels_json: path to the fgrained labels JSON file
    - antenna_api_base_url: base URL for the Antenna API
    - antenna_api_token: API token for the Antenna API
    - antenna_taxa_cache_json: path to the antenna taxa cache JSON file

    Returns:
    - taxon_to_idx: dict of gbif_taxon_key (str) -> Quebec class index
    """
    quebec_map = load_quebec_category_map(quebec_category_map_json)
    fgrained_path = fgrained_labels_json or DEFAULT_FGRAINED_LABELS

    if manifest_csv and Path(manifest_csv).exists():
        df = pd.read_csv(manifest_csv)
        df = df[df["status"] == "kept"]
        if "gbif_taxon_key" not in df.columns:
            raise ValueError(
                "manifest.csv missing gbif_taxon_key column; re-run convert_csv_export_to_ml_dataset.py"
            )
        missing = df["gbif_taxon_key"].isna()
        if missing.any():
            raise ValueError(
                f"manifest.csv has {int(missing.sum())} kept rows with null gbif_taxon_key"
            )
        species_col = "determination_name"
        taxon_col = "gbif_taxon_key"
    elif atlantic_csv:
        trainable_df, _ = filter_atlantic_trainable_rows(
            pd.read_csv(atlantic_csv), quebec_map
        )
        cache_path = (
            Path(antenna_taxa_cache_json)
            if antenna_taxa_cache_json
            else Path(output_path).parent / "antenna_taxon_cache.json"
        )
        client = AntennaTaxonClient(
            base_url=antenna_api_base_url,
            api_token=antenna_api_token,
            cache_path=cache_path,
        )
        df = resolve_gbif_taxon_keys_from_dataframe(
            trainable_df,
            client=client,
            fgrained_labels_path=fgrained_path,
            issues_report_path=Path(output_path).parent / "antenna_taxon_issues.csv",
        )
        species_col = "determination_name"
        taxon_col = "gbif_taxon_key"
    else:
        raise ValueError("Provide either --manifest-csv or --atlantic-csv")

    taxon_to_idx: dict[str, int] = {}
    for _, row in df.iterrows():
        name = row[species_col]
        taxon_key = str(int(float(row[taxon_col])))
        if name in quebec_map:
            taxon_to_idx[taxon_key] = quebec_map[name]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(taxon_to_idx, f, indent=2)

    print(f"Wrote {len(taxon_to_idx)} taxa to {out}", flush=True)
    return taxon_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Atlantic Forestry GBIF taxon key -> Quebec class index map."
    )
    parser.add_argument("--atlantic-csv", default=DEFAULT_ATLANTIC_CSV)
    parser.add_argument("--manifest-csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--quebec-category-map-json", default=DEFAULT_QUEBEC_MAP)
    parser.add_argument("--fgrained-labels-json", default=DEFAULT_FGRAINED_LABELS)
    parser.add_argument("--antenna-api-base-url", default=DEFAULT_ANTENNA_API_BASE_URL)
    parser.add_argument("--antenna-api-token", default=None)
    parser.add_argument("--antenna-taxa-cache-json", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_atlantic_category_map(
        quebec_category_map_json=args.quebec_category_map_json,
        output_path=args.output,
        atlantic_csv=args.atlantic_csv,
        manifest_csv=args.manifest_csv if Path(args.manifest_csv).exists() else None,
        fgrained_labels_json=args.fgrained_labels_json,
        antenna_api_base_url=args.antenna_api_base_url,
        antenna_api_token=args.antenna_api_token,
        antenna_taxa_cache_json=args.antenna_taxa_cache_json,
    )
