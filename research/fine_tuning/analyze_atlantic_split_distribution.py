#!/usr/bin/env python
# coding: utf-8

"""
Summarize per-species image counts in Atlantic Forestry train/val splits.

Reads `train.csv` and `val.csv` (columns: `taxonkey`, `filename`), maps GBIF keys to
species names via `manifest.csv`, and writes a distribution table plus a short report.

Example:

    python research/fine_tuning/analyze_atlantic_split_distribution.py \
      --data-dir ~/vanessa/data/fine_tuning_data_atlantic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from atlantic_dataset_utils import load_quebec_category_map, normalize_taxon_key



DEFAULT_DATA_DIR = "~/vanessa/data/fine_tuning_data_atlantic"
DEFAULT_QUEBEC_MAP = "~/vanessa/data/models/quebec-vermont_moth-category-map_19Jan2023.json"
TOP_K_COVERAGE_MAX = 7


def _compute_top_k_coverage(pivot: pd.DataFrame, max_k: int = TOP_K_COVERAGE_MAX) -> list[dict]:
    """Share of train/val images covered by the top-k species ranked by total image count."""
    ranked = pivot.sort_values(["n_total", "species_name"], ascending=[False, True])
    n_train_total = int(pivot["n_train"].sum())
    n_val_total = int(pivot["n_val"].sum())
    n_total = int(pivot["n_total"].sum())

    coverage: list[dict] = []
    for k in range(1, min(max_k, len(ranked)) + 1):
        top_k = ranked.head(k)
        n_train_top = int(top_k["n_train"].sum())
        n_val_top = int(top_k["n_val"].sum())
        n_total_top = int(top_k["n_total"].sum())
        coverage.append(
            {
                "top_k": k,
                "species_names": top_k["species_name"].tolist(),
                "n_train_images": n_train_top,
                "n_val_images": n_val_top,
                "n_total_images": n_total_top,
                "train_fraction": n_train_top / n_train_total if n_train_total else 0.0,
                "val_fraction": n_val_top / n_val_total if n_val_total else 0.0,
                "total_fraction": n_total_top / n_total if n_total else 0.0,
            }
        )
    return coverage


def _load_split_csv(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} split: {path}")
    df = pd.read_csv(path)
    required = {"taxonkey", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["taxonkey"] = df["taxonkey"].apply(normalize_taxon_key)
    df = df.dropna(subset=["taxonkey"])
    df["split"] = split_name
    return df


def _build_taxon_to_species_from_manifest(manifest_csv: Path) -> dict[str, str]:
    manifest = pd.read_csv(manifest_csv)
    kept = manifest[manifest["status"] == "kept"].copy()
    if kept.empty:
        raise ValueError(f"No kept rows in manifest: {manifest_csv}")
    if "gbif_taxon_key" not in kept.columns or "determination_name" not in kept.columns:
        raise ValueError(
            f"{manifest_csv} must include gbif_taxon_key and determination_name columns"
        )

    kept["gbif_taxon_key"] = kept["gbif_taxon_key"].apply(normalize_taxon_key)
    kept = kept.dropna(subset=["gbif_taxon_key", "determination_name"])

    taxon_to_species: dict[str, str] = {}
    for taxon_key, group in kept.groupby("gbif_taxon_key"):
        names = group["determination_name"].dropna().unique()
        if len(names) > 1:
            raise ValueError(
                f"Manifest maps taxon key {taxon_key} to multiple species names: {names.tolist()}"
            )
        taxon_to_species[taxon_key] = names[0]
    return taxon_to_species


def _build_taxon_to_species_from_bridge(
    bridge_json: Path, quebec_map_json: Path
) -> dict[str, str]:
    with open(bridge_json, "r", encoding="utf-8") as f:
        taxon_to_idx = json.load(f)
    quebec_map = load_quebec_category_map(quebec_map_json)
    idx_to_name = {idx: name for name, idx in quebec_map.items()}

    taxon_to_species: dict[str, str] = {}
    for taxon_key, class_idx in taxon_to_idx.items():
        species_name = idx_to_name.get(class_idx)
        if species_name is None:
            raise ValueError(
                f"Class index {class_idx} for taxon key {taxon_key} not found in Quebec map"
            )
        taxon_to_species[normalize_taxon_key(taxon_key)] = species_name
    return taxon_to_species


def build_taxon_to_species(
    manifest_csv: str | None,
    bridge_json: str | None,
    quebec_map_json: str,
) -> dict[str, str]:
    manifest_path = Path(manifest_csv).expanduser() if manifest_csv else None
    if manifest_path and manifest_path.exists():
        return _build_taxon_to_species_from_manifest(manifest_path)

    bridge_path = Path(bridge_json).expanduser() if bridge_json else None
    if bridge_path and bridge_path.exists():
        return _build_taxon_to_species_from_bridge(
            bridge_path, Path(quebec_map_json).expanduser()
        )

    tried = [str(p) for p in (manifest_path, bridge_path) if p is not None]
    raise FileNotFoundError(
        "Could not resolve species names. Expected one of:\n"
        + "\n".join(f"  - {path}" for path in tried)
    )


def build_split_distribution(
    data_dir: str,
    manifest_csv: str | None = None,
    bridge_json: str | None = None,
    quebec_map_json: str = DEFAULT_QUEBEC_MAP,
    output_dir: str | None = None,
) -> pd.DataFrame:
    data_path = Path(data_dir).expanduser()
    train_df = _load_split_csv(data_path / "train.csv", "train")
    val_df = _load_split_csv(data_path / "val.csv", "val")
    splits = pd.concat([train_df, val_df], ignore_index=True)

    taxon_to_species = build_taxon_to_species(
        manifest_csv=manifest_csv or str(data_path / "manifest.csv"),
        bridge_json=bridge_json or str(data_path / "taxon_to_training_idx.json"),
        quebec_map_json=quebec_map_json,
    )

    unknown_keys = sorted(set(splits["taxonkey"]) - set(taxon_to_species))
    if unknown_keys:
        raise ValueError(
            f"{len(unknown_keys)} taxon keys in splits are missing from species lookup: "
            f"{unknown_keys[:5]}{'...' if len(unknown_keys) > 5 else ''}"
        )

    splits["species_name"] = splits["taxonkey"].map(taxon_to_species)

    counts = (
        splits.groupby(["taxonkey", "species_name", "split"], as_index=False)
        .size()
        .rename(columns={"size": "n_images"})
    )
    pivot = counts.pivot_table(
        index=["taxonkey", "species_name"],
        columns="split",
        values="n_images",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for col in ("train", "val"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.rename(columns={"train": "n_train", "val": "n_val"})
    pivot["n_total"] = pivot["n_train"] + pivot["n_val"]
    pivot["train_fraction"] = pivot["n_train"] / pivot["n_total"]
    pivot["val_fraction"] = pivot["n_val"] / pivot["n_total"]

    quebec_map = load_quebec_category_map(quebec_map_json)
    pivot["quebec_class_idx"] = pivot["species_name"].map(quebec_map)
    pivot = pivot.sort_values(["n_total", "species_name"], ascending=[False, True])
    pivot = pivot.reset_index(drop=True)

    out_dir = Path(output_dir or data_path / "split_analysis").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out_dir / "species_split_distribution.csv", index=False)

    species_val_only_df = pivot.loc[pivot["n_train"] == 0].sort_values(
        ["n_total", "species_name"], ascending=[False, True]
    )
    max_total = int(pivot["n_total"].max())
    min_total = int(pivot["n_total"].min())
    highest_count_species = pivot.loc[pivot["n_total"] == max_total].sort_values(
        "species_name"
    )
    lowest_count_species = pivot.loc[pivot["n_total"] == min_total].sort_values(
        "species_name"
    )
    top_k_coverage = _compute_top_k_coverage(pivot)

    summary = {
        "n_species": int(pivot["species_name"].nunique()),
        "n_train_images": int(pivot["n_train"].sum()),
        "n_val_images": int(pivot["n_val"].sum()),
        "n_total_images": int(pivot["n_total"].sum()),
        "train_fraction_overall": float(pivot["n_train"].sum() / pivot["n_total"].sum()),
        "val_fraction_overall": float(pivot["n_val"].sum() / pivot["n_total"].sum()),
        "min_images_per_species": min_total,
        "max_images_per_species": max_total,
        "median_images_per_species": float(pivot["n_total"].median()),
        "species_train_only": sorted(
            pivot.loc[pivot["n_val"] == 0, "species_name"].tolist()
        ),
        "species_val_only": species_val_only_df["species_name"].tolist(),
        "highest_count_species": [
            {
                "species_name": row["species_name"],
                "taxonkey": row["taxonkey"],
                "n_train": int(row["n_train"]),
                "n_val": int(row["n_val"]),
                "n_total": int(row["n_total"]),
            }
            for _, row in highest_count_species.iterrows()
        ],
        "lowest_count_species": [
            {
                "species_name": row["species_name"],
                "taxonkey": row["taxonkey"],
                "n_train": int(row["n_train"]),
                "n_val": int(row["n_val"]),
                "n_total": int(row["n_total"]),
            }
            for _, row in lowest_count_species.iterrows()
        ],
        "top_species_coverage": top_k_coverage,
    }
    coverage_df = pd.DataFrame(top_k_coverage)
    coverage_df["species_names"] = coverage_df["species_names"].apply(
        lambda names: "; ".join(names)
    )
    coverage_df.to_csv(out_dir / "top_species_coverage.csv", index=False)
    with open(out_dir / "species_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def _species_table(rows: pd.DataFrame) -> list[str]:
        lines = [
            "| Species | GBIF key | Train | Val | Total | Train % |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['species_name']} | {row['taxonkey']} | "
                f"{int(row['n_train'])} | {int(row['n_val'])} | {int(row['n_total'])} | "
                f"{row['train_fraction']:.1%} |"
            )
        return lines

    report_lines = [
        "# Atlantic Forestry train/val split distribution",
        "",
        f"- Species: {summary['n_species']}",
        f"- Train images: {summary['n_train_images']} ({summary['train_fraction_overall']:.1%})",
        f"- Val images: {summary['n_val_images']} ({summary['val_fraction_overall']:.1%})",
        f"- Images per species: min {summary['min_images_per_species']}, "
        f"median {summary['median_images_per_species']:.0f}, "
        f"max {summary['max_images_per_species']}",
        "",
        "## Top-k species coverage (ranked by total image count)",
        "",
        "Share of images from the top 1–7 species by total count.",
        "",
        "| Top k | Train % | Val % | Total % | Train images | Val images |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_k_coverage:
        report_lines.append(
            f"| {row['top_k']} | {row['train_fraction']:.1%} | {row['val_fraction']:.1%} | "
            f"{row['total_fraction']:.1%} | {row['n_train_images']} | {row['n_val_images']} |"
        )
    report_lines.extend(
        [
            "",
            f"## Highest image counts ({max_total} images)",
            "",
            *_species_table(highest_count_species),
            "",
            f"## Lowest image counts ({min_total} images)",
            "",
            *_species_table(lowest_count_species),
            "",
            "## Species with no train images",
            f"- Count: {len(summary['species_val_only'])}",
            "",
        ]
    )
    if species_val_only_df.empty:
        report_lines.append("_None._")
    else:
        report_lines.extend(_species_table(species_val_only_df))
    report_lines.extend(
        [
            "",
            "## Species with no val images",
            f"- Count: {len(summary['species_train_only'])}",
            "",
            "## Top 10 species by total image count",
            "",
            "| Species | GBIF key | Train | Val | Total | Train % |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in pivot.head(10).iterrows():
        report_lines.append(
            f"| {row['species_name']} | {row['taxonkey']} | "
            f"{int(row['n_train'])} | {int(row['n_val'])} | {int(row['n_total'])} | "
            f"{row['train_fraction']:.1%} |"
        )
    (out_dir / "species_split_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote distribution table to {out_dir / 'species_split_distribution.csv'}", flush=True)
    print(f"Wrote top-k coverage to {out_dir / 'top_species_coverage.csv'}", flush=True)
    print(f"Wrote summary to {out_dir / 'species_split_summary.json'}", flush=True)
    print(f"Wrote report to {out_dir / 'species_split_report.md'}", flush=True)
    print("", flush=True)
    print(
        f"{summary['n_species']} species, "
        f"{summary['n_train_images']} train + {summary['n_val_images']} val images",
        flush=True,
    )
    for row in top_k_coverage:
        print(
            f"  Top {row['top_k']}: train {row['train_fraction']:.1%}, "
            f"val {row['val_fraction']:.1%}, total {row['total_fraction']:.1%}",
            flush=True,
        )
    return pivot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-species counts in Atlantic Forestry train/val splits."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Path to manifest.csv (default: {data_dir}/manifest.csv)",
    )
    parser.add_argument(
        "--bridge-json",
        default=None,
        help="Path to taxon_to_training_idx.json (default: {data_dir}/taxon_to_training_idx.json)",
    )
    parser.add_argument("--quebec-category-map-json", default=DEFAULT_QUEBEC_MAP)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: {data_dir}/split_analysis)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_split_distribution(
        data_dir=args.data_dir,
        manifest_csv=args.manifest_csv,
        bridge_json=args.bridge_json,
        quebec_map_json=args.quebec_category_map_json,
        output_dir=args.output_dir,
    )
