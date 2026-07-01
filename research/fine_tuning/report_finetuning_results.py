#!/usr/bin/env python
# coding: utf-8

"""
Summarize val_loss from W&B, evaluate baseline + fine-tuned checkpoints, compare.

1. Best-epoch val_loss per run is used for model selection
2. Run offline AMI-Traps evaluate_model on the Quebec baseline and each fine-tuned checkpoint
3. Run compare_evaluations baseline vs each fine-tuned run. This produces per-species
comparison and summary reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.fine_tuning.evaluation import compare_evaluations, evaluate_model

DEFAULT_WANDB_ENTITY = "vanessavaleriemac-mila"
DEFAULT_WANDB_PROJECT = "atlantic-forestry"
DEFAULT_RUN_NAME_PREFIX = "atlantic-forestry_lr"
DEFAULT_SWEEP_SUFFIX = "_30ep"
DEFAULT_EXPECTED_LRS = ("1e-3", "5e-4", "3e-4", "1e-4")
DEFAULT_BASELINE_MODEL = (
    "~/data/models/moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth"
)
DEFAULT_OUTPUT_DIR = "~/data/fine_tuning_data_atlantic/eval/lr_sweep"
DEFAULT_DATA_DIR = "~/data/fine_tuning_data"
DEFAULT_TEST_WEBDATASET = (
    "~/data/fine_tuning_data/webdataset/test/test-{000000..000003}.tar"
)
DEFAULT_CATEGORY_MAP = "~/data/models/quebec-vermont_moth-category-map_19Jan2023.json"
DEFAULT_OVERLAP_TABLE = (
    "~/data/fine_tuning_data_atlantic/overlap_analysis/species_overlap_table.csv"
)
DEFAULT_ATLANTIC_COUNTS = (
    "~/data/fine_tuning_data_atlantic/overlap_analysis/species_counts_atlantic.csv"
)
DEFAULT_SPLIT_SUMMARY = (
    "~/data/fine_tuning_data_atlantic/split_analysis/species_split_summary.json"
)
DEFAULT_SPLIT_DISTRIBUTION = (
    "~/data/fine_tuning_data_atlantic/split_analysis/species_split_distribution.csv"
)
TOP_ATLANTIC_SPECIES_K = 10
AMI_TRAPS_ONLY_HIGHLIGHT_K = 5
NUM_CLASSES = 3107
VAL_LOSS_TOLERANCE = 1e-4


@dataclass
class SweepRunRecord:
    run_name: str
    learning_rate: float | str
    best_epoch: int | None
    best_val_loss: float
    checkpoint_path: str
    wandb_run_url: str
    checkpoint_val_loss: float | None
    checkpoint_epoch: int | None


def _parse_learning_rates(values: list[str]) -> list[str]:
    return values if values else list(DEFAULT_EXPECTED_LRS)


def _format_lr(lr: float | str) -> str:
    if isinstance(lr, str):
        try:
            return f"{float(lr):g}"
        except ValueError:
            return lr
    return f"{lr:g}"


def _best_val_loss_from_history(run) -> tuple[float, int | None]:
    history = run.history(samples=10_000)
    if history.empty or "val_loss" not in history.columns:
        summary_loss = run.summary.get("val_loss")
        if summary_loss is not None:
            return float(summary_loss), run.summary.get("epoch")
        raise ValueError(f"No val_loss history for run {run.name}")

    valid = history.dropna(subset=["val_loss"])
    if valid.empty:
        raise ValueError(f"No non-null val_loss values for run {run.name}")

    best_idx = valid["val_loss"].idxmin()
    best_row = valid.loc[best_idx]
    best_epoch = None
    for epoch_col in ("epoch", "_step"):
        if epoch_col in best_row and pd.notna(best_row[epoch_col]):
            best_epoch = int(best_row[epoch_col])
            break
    return float(best_row["val_loss"]), best_epoch


def _download_model_checkpoint(run, cache_dir: Path) -> Path:
    artifacts = list(run.logged_artifacts())
    model_artifacts = [artifact for artifact in artifacts if artifact.type == "model"]
    if not model_artifacts:
        raise FileNotFoundError(f"No model artifact logged for W&B run {run.name}")

    artifact = model_artifacts[0]
    download_root = cache_dir / run.name
    download_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact.download(root=str(download_root)))

    checkpoint_files = sorted(artifact_dir.rglob("*_checkpoint.pt"))
    if not checkpoint_files:
        checkpoint_files = sorted(artifact_dir.rglob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No .pt checkpoint found in artifact for W&B run {run.name} ({artifact_dir})"
        )
    return checkpoint_files[0]


def _load_checkpoint_metadata(checkpoint_path: Path) -> tuple[float | None, int | None]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    val_loss = payload.get("val_loss")
    epoch = payload.get("epoch")
    return (
        float(val_loss) if val_loss is not None else None,
        int(epoch) if epoch is not None else None,
    )


def _run_matches_name_filters(
    run_name: str,
    *,
    run_name_suffix: str | None,
    exclude_run_name_suffix: str | None,
) -> bool:
    if run_name_suffix and run_name_suffix not in run_name:
        return False
    if exclude_run_name_suffix and exclude_run_name_suffix in run_name:
        return False
    return True


def _default_output_dir_for_suffix(run_name_suffix: str | None) -> str:
    if not run_name_suffix:
        return DEFAULT_OUTPUT_DIR
    tag = run_name_suffix.lstrip("_")
    return f"{DEFAULT_OUTPUT_DIR}_{tag}"


def fetch_sweep_runs_from_wandb(
    entity: str,
    project: str,
    run_name_prefix: str,
    checkpoint_cache_dir: Path,
    run_names_filter: set[str] | None = None,
    run_name_suffix: str | None = None,
    exclude_run_name_suffix: str | None = None,
) -> list[SweepRunRecord]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}"
    records: list[SweepRunRecord] = []

    for run in api.runs(path):
        if run.state == "crashed":
            continue
        if not run.name or not run.name.startswith(run_name_prefix):
            continue
        if run_names_filter and run.name not in run_names_filter:
            continue
        if not _run_matches_name_filters(
            run.name,
            run_name_suffix=run_name_suffix,
            exclude_run_name_suffix=exclude_run_name_suffix,
        ):
            continue

        best_val_loss, best_epoch = _best_val_loss_from_history(run)
        checkpoint_path = _download_model_checkpoint(run, checkpoint_cache_dir)
        ckpt_val_loss, ckpt_epoch = _load_checkpoint_metadata(checkpoint_path)

        if (
            ckpt_val_loss is not None
            and abs(ckpt_val_loss - best_val_loss) > VAL_LOSS_TOLERANCE
        ):
            print(
                f"Warning: W&B val_loss ({best_val_loss:.6f}) != checkpoint val_loss "
                f"({ckpt_val_loss:.6f}) for {run.name}",
                flush=True,
            )

        learning_rate = run.config.get("learning_rate", "unknown")
        records.append(
            SweepRunRecord(
                run_name=run.name,
                learning_rate=learning_rate,
                best_epoch=ckpt_epoch if ckpt_epoch is not None else best_epoch,
                best_val_loss=(
                    ckpt_val_loss if ckpt_val_loss is not None else best_val_loss
                ),
                checkpoint_path=str(checkpoint_path),
                wandb_run_url=run.url,
                checkpoint_val_loss=ckpt_val_loss,
                checkpoint_epoch=ckpt_epoch,
            )
        )

    records.sort(key=lambda row: row.best_val_loss)
    return records


def _records_to_dataframe(records: list[SweepRunRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_name": row.run_name,
                "learning_rate": row.learning_rate,
                "best_epoch": row.best_epoch,
                "best_val_loss": row.best_val_loss,
                "checkpoint_path": row.checkpoint_path,
                "wandb_run_url": row.wandb_run_url,
                "checkpoint_val_loss": row.checkpoint_val_loss,
                "checkpoint_epoch": row.checkpoint_epoch,
            }
            for row in records
        ]
    )


def write_val_loss_summary(
    records: list[SweepRunRecord],
    output_dir: Path,
    expected_learning_rates: list[str],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _records_to_dataframe(records)
    df.to_csv(output_dir / "val_loss_summary.csv", index=False)

    found_lrs = {_format_lr(row.learning_rate) for row in records}
    missing_lrs = [
        lr for lr in expected_learning_rates if _format_lr(lr) not in found_lrs
    ]
    if missing_lrs:
        print(
            f"Warning: missing W&B runs for learning rates: {missing_lrs}", flush=True
        )

    best = records[0] if records else None
    lines = [
        "# LR sweep — val_loss at best epoch",
        "",
        "Model selection uses **lowest Atlantic val_loss** (methods.md). "
        "AMI-Traps metrics below are reporting only.",
        "",
    ]
    if best:
        lines.extend(
            [
                f"**Best run (lowest val_loss):** `{best.run_name}` "
                f"(lr={best.learning_rate}, val_loss={best.best_val_loss:.6f})",
                "",
            ]
        )
    lines.append("| run_name | learning_rate | best_epoch | best_val_loss |")
    lines.append("|----------|---------------|------------|---------------|")
    for row in records:
        epoch = row.best_epoch if row.best_epoch is not None else ""
        lines.append(
            f"| {row.run_name} | {row.learning_rate} | {epoch} | {row.best_val_loss:.6f} |"
        )
    lines.append("")
    (output_dir / "val_loss_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n## val_loss at best epoch (ranked)", flush=True)
    print(df.to_string(index=False), flush=True)
    if best:
        print(
            f"\nBest run for model selection: {best.run_name} "
            f"(val_loss={best.best_val_loss:.6f})",
            flush=True,
        )
    return df


def _read_summary_metrics(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_acc(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None and pd.notna(value) else "—"


def _format_delta(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.2f} pp"


def _load_atlantic_species_subsets(
    atlantic_counts_csv: Path,
    split_summary_json: Path,
    top_k: int = TOP_ATLANTIC_SPECIES_K,
) -> tuple[list[str], dict[str, int], list[str]]:
    if not atlantic_counts_csv.exists():
        raise FileNotFoundError(f"Atlantic species counts not found: {atlantic_counts_csv}")
    if not split_summary_json.exists():
        raise FileNotFoundError(f"Split summary not found: {split_summary_json}")

    atlantic_counts = pd.read_csv(atlantic_counts_csv).sort_values(
        "n_images", ascending=False
    )
    top_species = atlantic_counts.head(top_k)["species_name"].tolist()
    atlantic_n = dict(
        zip(atlantic_counts["species_name"], atlantic_counts["n_images"].astype(int))
    )

    with open(split_summary_json, "r", encoding="utf-8") as f:
        split_summary = json.load(f)
    no_train_species = split_summary.get("species_val_only", [])
    return top_species, atlantic_n, no_train_species


def _load_split_distribution_maps(
    split_distribution_csv: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    if not split_distribution_csv.exists():
        return {}, {}
    split_df = pd.read_csv(split_distribution_csv)
    train_n = dict(zip(split_df["species_name"], split_df["n_train"].astype(int)))
    val_n = dict(zip(split_df["species_name"], split_df["n_val"].astype(int)))
    return train_n, val_n


def _comparison_row_for_species(
    comparison: pd.DataFrame, species_name: str
) -> pd.Series | None:
    rows = comparison.loc[comparison["species"] == species_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _subset_change_counts(comparison: pd.DataFrame, species_names: list[str]) -> dict[str, int]:
    counts = {"improved": 0, "regressed": 0, "unchanged": 0, "missing": 0}
    for species in species_names:
        row = _comparison_row_for_species(comparison, species)
        if row is None:
            counts["missing"] += 1
            continue
        change = row.get("top1_change", "unknown")
        if change in counts:
            counts[change] += 1
        else:
            counts["missing"] += 1
    return counts


def _render_species_subset_table(
    comparison: pd.DataFrame,
    species_names: list[str],
    *,
    atlantic_n: dict[str, int],
    atlantic_n_label: str,
    val_n: dict[str, int] | None = None,
) -> list[str]:
    lines = [
        f"| Species | {atlantic_n_label} | AMI-Traps test n | "
        "Baseline top-1 | Finetuned top-1 | Status | Δ top-1 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for species in species_names:
        row = _comparison_row_for_species(comparison, species)
        atlantic_count = atlantic_n.get(species)
        if val_n is not None and species in val_n:
            atlantic_count = val_n[species]
        atlantic_str = str(atlantic_count) if atlantic_count is not None else "—"

        if row is None:
            lines.append(
                f"| {species} | {atlantic_str} | — | — | — | not in AMI-Traps test | — |"
            )
            continue

        ami_traps_n = row.get("n_baseline")
        if pd.isna(ami_traps_n):
            ami_traps_n = row.get("n")
        ami_traps_str = str(int(ami_traps_n)) if pd.notna(ami_traps_n) else "—"
        lines.append(
            f"| {species} | {atlantic_str} | {ami_traps_str} | "
            f"{_format_acc(row.get('top1_acc_baseline'))} | "
            f"{_format_acc(row.get('top1_acc_finetuned'))} | "
            f"{row.get('top1_change', 'unknown')} | "
            f"{_format_delta(row.get('delta_top1_acc'))} |"
        )
    return lines


def _render_ami_traps_only_highlights(ami_traps_only: pd.DataFrame) -> list[str]:
    if ami_traps_only.empty:
        return ["_No AMI-Traps-only species in comparison._", ""]

    valid = ami_traps_only.dropna(subset=["delta_top1_acc"]).copy()
    improved = valid.loc[valid["top1_change"] == "improved"].sort_values(
        "delta_top1_acc", ascending=False
    )
    regressed = valid.loc[valid["top1_change"] == "regressed"].sort_values(
        "delta_top1_acc", ascending=True
    )

    lines = [
        f"- Species in subset: {len(ami_traps_only)}",
        f"- Improved: {len(improved)}",
        f"- Regressed: {len(regressed)}",
        f"- Unchanged: {(valid['top1_change'] == 'unchanged').sum()}",
        "",
    ]

    def _highlight_table(title: str, subset: pd.DataFrame) -> list[str]:
        if subset.empty:
            return [f"### {title}", "", "_None._", ""]
        best = subset.iloc[0]
        section = [
            f"### {title}",
            "",
            f"**Largest change:** {best['species']} "
            f"({_format_delta(best['delta_top1_acc'])}, "
            f"{_format_acc(best['top1_acc_baseline'])} → "
            f"{_format_acc(best['top1_acc_finetuned'])}, "
            f"n={int(best['n_baseline'])})",
            "",
            "| Species | AMI-Traps test n | Baseline top-1 | Finetuned top-1 | Δ top-1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in subset.head(AMI_TRAPS_ONLY_HIGHLIGHT_K).iterrows():
            section.append(
                f"| {row['species']} | {int(row['n_baseline'])} | "
                f"{_format_acc(row['top1_acc_baseline'])} | "
                f"{_format_acc(row['top1_acc_finetuned'])} | "
                f"{_format_delta(row['delta_top1_acc'])} |"
            )
        section.append("")
        return section

    lines.extend(_highlight_table("Most improved", improved))
    lines.extend(_highlight_table("Most regressed", regressed))
    return lines


def write_species_subset_report(
    records: list[SweepRunRecord],
    output_dir: Path,
    *,
    atlantic_counts_csv: str,
    split_summary_json: str,
    split_distribution_csv: str,
    top_k: int = TOP_ATLANTIC_SPECIES_K,
) -> Path:
    """Write per-run species subset comparison report next to val_loss_summary.md."""
    atlantic_counts_path = Path(atlantic_counts_csv).expanduser()
    split_summary_path = Path(split_summary_json).expanduser()
    split_distribution_path = Path(split_distribution_csv).expanduser()

    top_species, atlantic_n, no_train_species = _load_atlantic_species_subsets(
        atlantic_counts_path, split_summary_path, top_k=top_k
    )
    _, val_n = _load_split_distribution_maps(split_distribution_path)

    lines = [
        "# LR sweep — species subset comparison",
        "",
        "Per-species AMI-Traps test comparison vs Quebec baseline. "
        "Atlantic image counts come from overlap analysis and split distribution.",
        "",
        f"- Top {top_k} Atlantic species source: `{atlantic_counts_path}`",
        f"- No-train species source: `{split_summary_path}` (`species_val_only`)",
        "",
    ]

    runs_with_comparisons = 0
    for record in records:
        comparison_dir = output_dir / f"comparison_{record.run_name}"
        comparison_csv = comparison_dir / "per_species_comparison.csv"
        ami_traps_only_csv = comparison_dir / "ami_traps_only_comparison.csv"
        if not comparison_csv.exists():
            lines.extend(
                [
                    f"## {record.run_name}",
                    "",
                    "_Comparison artifacts not found; run evaluation with compare enabled._",
                    "",
                ]
            )
            continue

        runs_with_comparisons += 1
        comparison = pd.read_csv(comparison_csv)
        top_counts = _subset_change_counts(comparison, top_species)
        no_train_counts = _subset_change_counts(comparison, no_train_species)

        lines.extend(
            [
                f"## {record.run_name}",
                "",
                f"- Learning rate: {record.learning_rate}",
                f"- Best val_loss: {record.best_val_loss:.6f}",
                "",
                f"### Top {top_k} Atlantic species (by Atlantic image count)",
                "",
                f"- Improved: {top_counts['improved']}",
                f"- Regressed: {top_counts['regressed']}",
                f"- Unchanged: {top_counts['unchanged']}",
                f"- Not in AMI-Traps test: {top_counts['missing']}",
                "",
                *_render_species_subset_table(
                    comparison,
                    top_species,
                    atlantic_n=atlantic_n,
                    atlantic_n_label="Atlantic images",
                ),
                "",
                "### Species with no Atlantic train images",
                "",
                f"- Species: {len(no_train_species)}",
                f"- Improved: {no_train_counts['improved']}",
                f"- Regressed: {no_train_counts['regressed']}",
                f"- Unchanged: {no_train_counts['unchanged']}",
                f"- Not in AMI-Traps test: {no_train_counts['missing']}",
                "",
            ]
        )
        if no_train_species:
            lines.extend(
                _render_species_subset_table(
                    comparison,
                    no_train_species,
                    atlantic_n=atlantic_n,
                    atlantic_n_label="Atlantic val images",
                    val_n=val_n,
                )
            )
        else:
            lines.append("_None._")
        lines.append("")

        lines.extend(
            [
                "### AMI-Traps-only species",
                "",
            ]
        )
        if ami_traps_only_csv.exists():
            ami_traps_only = pd.read_csv(ami_traps_only_csv)
            lines.extend(_render_ami_traps_only_highlights(ami_traps_only))
        else:
            lines.append("_ami_traps_only_comparison.csv not found._")
            lines.append("")

    if runs_with_comparisons == 0:
        lines.extend(
            [
                "_No comparison artifacts found under this output directory. "
                "Re-run without `--skip-compare` after evaluations complete._",
                "",
            ]
        )

    report_path = output_dir / "species_subset_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote species subset report to {report_path}", flush=True)
    return report_path


def run_evaluations_and_comparisons(
    records: list[SweepRunRecord],
    *,
    output_dir: Path,
    baseline_model: str,
    data_dir: str,
    test_webdataset: str,
    category_map_json: str,
    overlap_table_csv: str | None,
    skip_baseline: bool,
    skip_finetuned: bool,
    skip_compare: bool,
) -> pd.DataFrame:
    baseline_dir = output_dir / "baseline"
    if not skip_baseline:
        evaluate_model(
            data_dir=data_dir,
            model_file=baseline_model,
            model_type="resnet50",
            num_classes=NUM_CLASSES,
            test_webdataset=test_webdataset,
            category_map_json=category_map_json,
            samples_type="all",
            run_name="baseline",
            output_dir=str(baseline_dir),
            checkpoint=False,
        )

    baseline_summary = _read_summary_metrics(baseline_dir / "summary.json")
    baseline_micro = baseline_summary.get("micro_top1_acc")
    baseline_macro = baseline_summary.get("macro_top1_acc")

    eval_rows = []
    for record in records:
        finetuned_dir = output_dir / record.run_name
        if not skip_finetuned:
            evaluate_model(
                data_dir=data_dir,
                model_file=record.checkpoint_path,
                model_type="resnet50",
                num_classes=NUM_CLASSES,
                test_webdataset=test_webdataset,
                category_map_json=category_map_json,
                samples_type="all",
                run_name=record.run_name,
                output_dir=str(finetuned_dir),
                checkpoint=True,
            )

        if not skip_compare and baseline_dir.exists() and finetuned_dir.exists():
            compare_evaluations(
                baseline_output_dir=str(baseline_dir),
                finetuned_output_dir=str(finetuned_dir),
                output_dir=str(output_dir / f"comparison_{record.run_name}"),
                overlap_table_csv=overlap_table_csv,
            )

        summary = _read_summary_metrics(finetuned_dir / "summary.json")
        micro = summary.get("micro_top1_acc")
        macro = summary.get("macro_top1_acc")
        eval_rows.append(
            {
                "run_name": record.run_name,
                "learning_rate": record.learning_rate,
                "best_val_loss": record.best_val_loss,
                "micro_top1_acc": micro,
                "macro_top1_acc": macro,
                "delta_micro_vs_baseline": (
                    None
                    if micro is None or baseline_micro is None
                    else micro - baseline_micro
                ),
                "delta_macro_vs_baseline": (
                    None
                    if macro is None or baseline_macro is None
                    else macro - baseline_macro
                ),
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    if not eval_df.empty:
        eval_df.to_csv(output_dir / "sweep_eval_summary.csv", index=False)
        print("\n## AMI-Traps eval summary (reporting only)", flush=True)
        print(eval_df.to_string(index=False), flush=True)
    return eval_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report LR sweep val_loss, evaluate checkpoints, compare to baseline."
    )
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--run-name-prefix", default=DEFAULT_RUN_NAME_PREFIX)
    parser.add_argument(
        "--run-name-suffix",
        default=None,
        help=(
            "Only include W&B runs whose name contains this substring "
            f"(e.g. {DEFAULT_SWEEP_SUFFIX!r} for learning_rate_sweep.sh). "
            "Also selects eval/lr_sweep_{tag} as the default output dir."
        ),
    )
    parser.add_argument(
        "--exclude-run-name-suffix",
        default=None,
        help="Exclude W&B runs whose name contains this substring (e.g. _30ep for the 10-epoch sweep).",
    )
    parser.add_argument(
        "--expected-learning-rates",
        nargs="+",
        default=list(DEFAULT_EXPECTED_LRS),
    )
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Report output directory. Defaults to eval/lr_sweep, or eval/lr_sweep_{tag} "
            "when --run-name-suffix is set."
        ),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--test-webdataset", default=DEFAULT_TEST_WEBDATASET)
    parser.add_argument("--category-map-json", default=DEFAULT_CATEGORY_MAP)
    parser.add_argument("--overlap-table-csv", default=DEFAULT_OVERLAP_TABLE)
    parser.add_argument("--atlantic-counts-csv", default=DEFAULT_ATLANTIC_COUNTS)
    parser.add_argument("--split-summary-json", default=DEFAULT_SPLIT_SUMMARY)
    parser.add_argument(
        "--split-distribution-csv", default=DEFAULT_SPLIT_DISTRIBUTION
    )
    parser.add_argument(
        "--skip-subset-report",
        action="store_true",
        help="Skip species_subset_report.md generation.",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="Optional subset of W&B run names to include.",
    )
    parser.add_argument(
        "--val-loss-only",
        action="store_true",
        help="Only fetch W&B val_loss summary; skip GPU evaluation.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name_suffix = args.run_name_suffix or None
    exclude_run_name_suffix = args.exclude_run_name_suffix or None
    output_dir = Path(
        args.output_dir or _default_output_dir_for_suffix(run_name_suffix)
    )
    checkpoint_cache_dir = output_dir / "wandb_checkpoints"
    run_names_filter = set(args.run_names) if args.run_names else None
    expected_lrs = _parse_learning_rates(args.expected_learning_rates)

    records = fetch_sweep_runs_from_wandb(
        entity=args.wandb_entity,
        project=args.wandb_project,
        run_name_prefix=args.run_name_prefix,
        checkpoint_cache_dir=checkpoint_cache_dir,
        run_names_filter=run_names_filter,
        run_name_suffix=run_name_suffix,
        exclude_run_name_suffix=exclude_run_name_suffix,
    )
    if not records:
        filters = [f"prefix {args.run_name_prefix!r}"]
        if run_name_suffix:
            filters.append(f"suffix containing {run_name_suffix!r}")
        if exclude_run_name_suffix:
            filters.append(f"excluding suffix {exclude_run_name_suffix!r}")
        raise SystemExit(
            f"No W&B runs found under {args.wandb_entity}/{args.wandb_project} "
            f"with {', '.join(filters)}"
        )

    write_val_loss_summary(records, output_dir, expected_lrs)

    if args.val_loss_only:
        print(f"\nWrote val_loss summary to {output_dir}", flush=True)
        return

    overlap_table = args.overlap_table_csv
    if overlap_table and not Path(overlap_table).exists():
        print(f"Warning: overlap table not found: {overlap_table}", flush=True)
        overlap_table = None

    run_evaluations_and_comparisons(
        records,
        output_dir=output_dir,
        baseline_model=args.baseline_model,
        data_dir=args.data_dir,
        test_webdataset=args.test_webdataset,
        category_map_json=args.category_map_json,
        overlap_table_csv=overlap_table,
        skip_baseline=args.skip_baseline,
        skip_finetuned=args.skip_finetuned,
        skip_compare=args.skip_compare,
    )

    if not args.skip_subset_report:
        for path_arg in (
            args.atlantic_counts_csv,
            args.split_summary_json,
        ):
            if not Path(path_arg).expanduser().exists():
                print(
                    f"Warning: species subset input not found: {path_arg}; "
                    "skipping species_subset_report.md",
                    flush=True,
                )
                break
        else:
            write_species_subset_report(
                records,
                output_dir,
                atlantic_counts_csv=args.atlantic_counts_csv,
                split_summary_json=args.split_summary_json,
                split_distribution_csv=args.split_distribution_csv,
            )

    print(f"\nDone. Outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
