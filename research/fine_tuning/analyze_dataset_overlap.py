#!/usr/bin/env python
# coding: utf-8

"""
Analyze species overlap between Atlantic Forestry fine-tuning data and AMI-Traps test.

Produces the following outputs:
- `species_counts_atlantic.csv` / `species_counts_ami_traps_test.csv`
- `species_overlap_table.csv` — includes `atlantic_gbif_key` and `ami_traps_gbif_key`
Must match for overlap species.
- `antenna_taxon_issues.csv` — Antenna taxa needing platform fixes
(see `recommended_action` column)
- `gbif_key_mismatches.csv` — Antenna vs fgrained disagreements aligned to fgrained
(for AMI-Traps folder/ID space; no longer a hard fail at resolve time)
- `species_overlap_summary.json` / `species_overlap_report.md`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from atlantic_dataset_utils import (
    DEFAULT_ANTENNA_API_BASE_URL,
    AntennaTaxonClient,
    GbifKeyMismatchError,
    aggregate_species_counts,
    build_taxon_to_quebec_idx,
    filter_atlantic_trainable_rows,
    load_quebec_category_map,
    normalize_taxon_key,
    print_gbif_mismatch_drop_report,
    resolve_gbif_taxon_keys_from_dataframe,
)

DEFAULT_ATLANTIC_CSV = "~/vanessa/data/exports/atlantic-forestry-centre_export-104.csv"
DEFAULT_AMI_TRAPS_TEST_CSV = "~/vanessa/data/fine_tuning_data/test.csv"
DEFAULT_FGRAINED_LABELS = "~/vanessa/data/ami_traps/insect_crops/fgrained_labels.json"
DEFAULT_QUEBEC_MAP = "~/vanessa/data/models/quebec-vermont_moth-category-map_19Jan2023.json"
DEFAULT_TAXON_TO_IDX = "~/vanessa/data/fine_tuning_data/taxon_to_quebec_idx.json"
DEFAULT_OUTPUT_DIR = "~/vanessa/data/fine_tuning_data_atlantic/overlap_analysis"


def _load_taxon_bridge(
    taxon_to_idx_path: str | None,
    fgrained_labels_path: str,
    quebec_map: dict[str, int],
) -> dict[str, int]:
    if taxon_to_idx_path and Path(taxon_to_idx_path).exists():
        with open(taxon_to_idx_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(fgrained_labels_path, "r", encoding="utf-8") as f:
        fgrained_labels = json.load(f)
    return build_taxon_to_quebec_idx(fgrained_labels, quebec_map)


def _idx_to_species_name(quebec_map: dict[str, int]) -> dict[int, str]:
    return {idx: name for name, idx in quebec_map.items()}


def _resolve_ami_traps_species_name(
    filename: str,
    taxon_key: str,
    fgrained_labels: dict,
    idx_to_name: dict[int, str],
    taxon_to_idx: dict[str, int],
) -> str | None:
    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg"):
        label_key = stem + ext
        if label_key in fgrained_labels:
            return fgrained_labels[label_key]["label"]

    class_idx = taxon_to_idx.get(taxon_key)
    if class_idx is not None:
        return idx_to_name.get(class_idx)
    return None


def _build_ami_traps_test_counts(
    test_csv_path: str,
    fgrained_labels_path: str,
    taxon_to_idx: dict[str, int],
    quebec_map: dict[str, int],
) -> pd.DataFrame:
    test_df = pd.read_csv(test_csv_path)
    test_df["taxonkey"] = test_df["taxonkey"].apply(normalize_taxon_key)
    test_df = test_df.dropna(subset=["taxonkey"])
    test_df = test_df[test_df["taxonkey"].isin(taxon_to_idx)]

    with open(fgrained_labels_path, "r", encoding="utf-8") as f:
        fgrained_labels = json.load(f)
    idx_to_name = _idx_to_species_name(quebec_map)

    species_names = []
    for _, row in test_df.iterrows():
        species_names.append(
            _resolve_ami_traps_species_name(
                row["filename"],
                row["taxonkey"],
                fgrained_labels,
                idx_to_name,
                taxon_to_idx,
            )
        )
    test_df["species_name"] = species_names
    test_df = test_df.dropna(subset=["species_name"])

    return aggregate_species_counts(test_df, "species_name", "taxonkey", quebec_map)


def _write_overlap_report(
    output_dir: Path,
    atlantic_counts: pd.DataFrame,
    ami_traps_counts: pd.DataFrame,
    overlap_species: set[str],
    atlantic_only: set[str],
    ami_traps_only: set[str],
    atlantic_exclusion_stats: dict[str, int],
) -> str:
    overlap_table = []
    overlap_mismatches: list[dict] = []
    for species in sorted(overlap_species):
        atl_row = atlantic_counts.loc[atlantic_counts["species_name"] == species]
        ami_row = ami_traps_counts.loc[ami_traps_counts["species_name"] == species]
        atlantic_key = str(atl_row["gbif_taxon_key"].iloc[0]) if len(atl_row) else None
        ami_traps_key = str(ami_row["gbif_taxon_key"].iloc[0]) if len(ami_row) else None
        n_atlantic = int(atl_row["n_images"].iloc[0]) if len(atl_row) else 0
        n_ami_traps_test = int(ami_row["n_images"].iloc[0]) if len(ami_row) else 0
        if atlantic_key != ami_traps_key:
            overlap_mismatches.append(
                {
                    "species_name": species,
                    "antenna_gbif_key": atlantic_key,
                    "fgrained_gbif_key": ami_traps_key,
                    "n_rows": n_atlantic,
                    "n_ami_traps_test": n_ami_traps_test,
                    "determination_id": "",
                    "resolution_source": "overlap_table",
                }
            )
            continue
        overlap_table.append(
            {
                "species_name": species,
                "atlantic_gbif_key": atlantic_key,
                "ami_traps_gbif_key": ami_traps_key,
                "n_atlantic": n_atlantic,
                "n_ami_traps_test": n_ami_traps_test,
                "quebec_class_idx": (
                    int(atl_row["quebec_class_idx"].iloc[0])
                    if len(atl_row)
                    else int(ami_row["quebec_class_idx"].iloc[0])
                ),
            }
        )
    if overlap_mismatches:
        print_gbif_mismatch_drop_report(
            overlap_mismatches,
            n_trainable_rows=int(atlantic_counts["n_images"].sum()),
            n_trainable_species=len(atlantic_counts),
        )
        pd.DataFrame(overlap_mismatches).to_csv(
            output_dir / "gbif_key_mismatches.csv", index=False
        )
        species_list = [m["species_name"] for m in overlap_mismatches]
        n_rows = sum(int(m["n_rows"]) for m in overlap_mismatches)
        raise GbifKeyMismatchError(
            f"{len(overlap_mismatches)} overlap GBIF key mismatch(es); "
            f"dropping them would exclude {n_rows} Atlantic rows / "
            f"{len(overlap_mismatches)} species: {species_list}",
            mismatches=overlap_mismatches,
        )
    overlap_df = pd.DataFrame(overlap_table)
    overlap_df.to_csv(output_dir / "species_overlap_table.csv", index=False)

    n_atlantic_images = int(atlantic_counts["n_images"].sum())
    n_ami_traps_images = int(ami_traps_counts["n_images"].sum())
    overlap_atlantic_images = (
        int(overlap_df["n_atlantic"].sum()) if len(overlap_df) else 0
    )
    overlap_ami_traps_images = (
        int(overlap_df["n_ami_traps_test"].sum()) if len(overlap_df) else 0
    )

    summary = {
        "atlantic_trainable": {
            "n_species": len(atlantic_counts),
            "n_images": n_atlantic_images,
        },
        "ami_traps_evaluable_test": {
            "n_species": len(ami_traps_counts),
            "n_images": n_ami_traps_images,
        },
        "overlap": {
            "n_species": len(overlap_species),
            "n_atlantic_images": overlap_atlantic_images,
            "n_ami_traps_test_images": overlap_ami_traps_images,
            "species": sorted(overlap_species),
        },
        "atlantic_only": {
            "n_species": len(atlantic_only),
            "species": sorted(atlantic_only),
        },
        "ami_traps_only": {
            "n_species": len(ami_traps_only),
            "species": sorted(ami_traps_only),
        },
        "atlantic_exclusion_stats": atlantic_exclusion_stats,
    }
    with open(output_dir / "species_overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Atlantic Forestry vs AMI-Traps species overlap",
        "",
        "## Atlantic Forestry (trainable, in Quebec map)",
        f"- Species: {len(atlantic_counts)}",
        f"- Images: {n_atlantic_images}",
        "",
        "## AMI-Traps test (evaluable webdataset population)",
        f"- Species: {len(ami_traps_counts)}",
        f"- Images: {n_ami_traps_images}",
        "",
        "## Overlap (candidate species for measurable AMI-Traps improvement)",
        f"- Species: {len(overlap_species)}",
        f"- Atlantic images: {overlap_atlantic_images}",
        f"- AMI-Traps test images: {overlap_ami_traps_images}",
        "",
        "## Atlantic-only species (fine-tuned but not in AMI-Traps test)",
        f"- Species: {len(atlantic_only)}",
        "",
        "## AMI-Traps-only species (in test eval, not seen during fine-tuning)",
        f"- Species: {len(ami_traps_only)}",
        "",
        "## Atlantic row exclusions",
    ]
    issues_path = output_dir / "antenna_taxon_issues.csv"
    if issues_path.exists():
        lines.extend(
            [
                "",
                "## Antenna taxon issues",
                f"- See `{issues_path.name}` for taxa needing platform fixes",
            ]
        )
    for key, value in atlantic_exclusion_stats.items():
        lines.append(f"- {key}: {value}")

    report = "\n".join(lines) + "\n"
    (output_dir / "species_overlap_report.md").write_text(report, encoding="utf-8")
    return report


def analyze_dataset_overlap(
    atlantic_csv: str,
    ami_traps_test_csv: str,
    fgrained_labels_json: str,
    quebec_category_map_json: str,
    output_dir: str,
    taxon_to_idx_json: str | None = None,
    antenna_api_base_url: str = DEFAULT_ANTENNA_API_BASE_URL,
    antenna_api_token: str | None = None,
    antenna_taxa_cache_json: str | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_path = (
        Path(antenna_taxa_cache_json)
        if antenna_taxa_cache_json
        else output_path.parent / "antenna_taxon_cache.json"
    )

    quebec_map = load_quebec_category_map(quebec_category_map_json)
    atlantic_df = pd.read_csv(atlantic_csv)
    trainable_df, exclusion_stats = filter_atlantic_trainable_rows(
        atlantic_df, quebec_map
    )
    client = AntennaTaxonClient(
        base_url=antenna_api_base_url,
        api_token=antenna_api_token,
        cache_path=cache_path,
    )
    trainable_df = resolve_gbif_taxon_keys_from_dataframe(
        trainable_df,
        client=client,
        fgrained_labels_path=fgrained_labels_json,
        issues_report_path=output_path / "antenna_taxon_issues.csv",
    )
    atlantic_counts = aggregate_species_counts(
        trainable_df, "determination_name", "gbif_taxon_key", quebec_map
    )
    atlantic_counts.to_csv(output_path / "species_counts_atlantic.csv", index=False)

    taxon_to_idx = _load_taxon_bridge(
        taxon_to_idx_json, fgrained_labels_json, quebec_map
    )
    ami_traps_counts = _build_ami_traps_test_counts(
        ami_traps_test_csv, fgrained_labels_json, taxon_to_idx, quebec_map
    )
    ami_traps_counts.to_csv(
        output_path / "species_counts_ami_traps_test.csv", index=False
    )

    atlantic_species = set(atlantic_counts["species_name"])
    ami_traps_species = set(ami_traps_counts["species_name"])
    overlap_species = atlantic_species & ami_traps_species
    atlantic_only = atlantic_species - ami_traps_species
    ami_traps_only = ami_traps_species - atlantic_species

    report = _write_overlap_report(
        output_path,
        atlantic_counts,
        ami_traps_counts,
        overlap_species,
        atlantic_only,
        ami_traps_only,
        exclusion_stats,
    )
    print(report, flush=True)
    return {
        "output_dir": str(output_path),
        "n_overlap_species": len(overlap_species),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze species overlap between Atlantic Forestry and AMI-Traps test."
    )
    parser.add_argument("--atlantic-csv", default=DEFAULT_ATLANTIC_CSV)
    parser.add_argument("--ami-traps-test-csv", default=DEFAULT_AMI_TRAPS_TEST_CSV)
    parser.add_argument("--fgrained-labels-json", default=DEFAULT_FGRAINED_LABELS)
    parser.add_argument("--quebec-category-map-json", default=DEFAULT_QUEBEC_MAP)
    parser.add_argument("--taxon-to-idx-json", default=DEFAULT_TAXON_TO_IDX)
    parser.add_argument("--antenna-api-base-url", default=DEFAULT_ANTENNA_API_BASE_URL)
    parser.add_argument("--antenna-api-token", default=None)
    parser.add_argument("--antenna-taxa-cache-json", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_dataset_overlap(
        atlantic_csv=args.atlantic_csv,
        ami_traps_test_csv=args.ami_traps_test_csv,
        fgrained_labels_json=args.fgrained_labels_json,
        quebec_category_map_json=args.quebec_category_map_json,
        output_dir=args.output_dir,
        taxon_to_idx_json=args.taxon_to_idx_json,
        antenna_api_base_url=args.antenna_api_base_url,
        antenna_api_token=args.antenna_api_token,
        antenna_taxa_cache_json=args.antenna_taxa_cache_json,
    )
