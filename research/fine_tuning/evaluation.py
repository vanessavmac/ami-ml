#!/usr/bin/env python
# coding: utf-8

"""
Evaluation of models on AMI Traps test set
Previously produced a single test accuracy = per-batch top-1 accuracy, averaged across batches.
Each batch counted equally regardless of number of samples.

Now, a micro top-1 accuracy (correct_samples / total_samples) means each image counts equally.
Common species influence the score more.

Macro top-1 accuracy computes a top-1 for each species, then takes unweighted mean across species.
This means each species counts equally, regardless of how many samples it has.

Use shared utilities in taxonomic_metrics.py to match fgrained model evaluation.
"""

import os
import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple

import dotenv
import pandas as pd
import torch

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.taxonomic_metrics import (
    EvaluationArtifacts,
    TaxonomicAccuracyState,
    batch_logits_to_predictions,
    compare_per_species_reports,
    load_id2label,
    print_species_accuracy_report,
    record_evaluation_sample,
    write_evaluation_artifacts,
)
from src.classification.utils import build_model

dotenv.load_dotenv()


def _format_species_change_list(df: pd.DataFrame, change: str) -> list[str]:
    """Format a bullet list of species with top-1 delta and AMI-Traps test n."""
    subset = df.loc[df["top1_change"] == change].copy()
    if subset.empty:
        return ["_None._"]

    ascending = change == "regressed"
    subset = subset.sort_values("delta_top1_acc", ascending=ascending, na_position="last")

    lines: list[str] = []
    for _, row in subset.iterrows():
        delta = row.get("delta_top1_acc")
        delta_str = f"{delta:+.2f} pp" if pd.notna(delta) else "—"
        n = row.get("n_baseline")
        if pd.isna(n):
            n = row.get("n")
        n_str = f", n={int(n)}" if pd.notna(n) else ""
        lines.append(f"- {row['species']} ({delta_str}{n_str})")
    return lines


def _filter_classes(
    data_dir: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    samples_type: Literal["all", "seen", "unseen"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter classes based on the samples type.

    Args:
        data_dir (str): Directory containing the dataset.
        images (torch.Tensor): Tensor of images.
        labels (torch.Tensor): Tensor of labels.
        samples_type (str): Type of samples to filter.

    Returns:
        [torch.Tensor, torch.Tensor]: Filtered labels and corresponding images.
    """
    if samples_type in ["seen", "unseen"]:
        # Load the species list
        with open(Path(data_dir) / f"{samples_type}_species.pkl", "rb") as f:
            species_list = pickle.load(f)

        # Create a mask for filtering
        mask = torch.isin(labels, torch.tensor(species_list).to(labels.device))

        # Filter the images and labels
        labels = labels[mask]
        images = images[mask]

    return images, labels


def evaluate_model(
    data_dir: str,
    model_file: str,
    model_type: str,
    num_classes: int,
    test_webdataset: str,
    category_map_json: str,
    samples_type: Literal["all", "seen", "unseen"],
    run_name: str = "evaluation",
    batch_size: int = 32,
    image_input_size: int = 128,
    preprocess_mode: str = "torch",
    output_dir: Optional[str] = None,
    checkpoint: bool = False,
    save_predictions: bool = False,
) -> EvaluationArtifacts:
    """Evaluate a model on the AMI Traps test set.

    Args:
        data_dir (str): Directory containing the dataset.
        model_file (str): Path to the model file.
        model_type (str): Type of the model (e.g., "convnext_tiny_in22k", "resnet50").
        num_classes (int): Number of output classes in the model.
        test_webdataset (str): Path to the test WebDataset file.
        category_map_json (str): Path to category map JSON (species name -> class index).
        samples_type (str): Type of samples to filter ("all", "seen", "unseen").
        run_name (str, optional): Label for the printed accuracy report.
        batch_size (int, optional): Batch size for evaluation. Defaults to 32.
        image_input_size (int, optional): Input size for the images. Defaults to 128.
        preprocess_mode (str, optional): Preprocessing mode for the images. Defaults to "torch".
        output_dir (Optional[str]): Optional directory to write per-species CSV/JSON artifacts.
        checkpoint (bool): If True, load model_state_dict from a training checkpoint .pt file.
        save_predictions (bool): If True, write per-image predictions_detail.csv.

    Returns:
        EvaluationArtifacts with accumulated metrics.
    """

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        device, model_type, num_classes, model_file, checkpoint=checkpoint
    )
    model.eval()

    # Species string to match fgrained evaluation, which uses shared utilities like check_prediction
    id2label = load_id2label(category_map_json)

    # Load the test dataset
    test_dataloader = build_webdataset_pipeline(
        test_webdataset,
        image_input_size,
        batch_size,
        preprocess_mode,
    )

    artifacts = EvaluationArtifacts(run_name=run_name, state=TaxonomicAccuracyState())

    # Iterate through the test dataset
    with torch.no_grad():
        for batch_data in test_dataloader:
            images, labels = batch_data
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            # Filter out required classes
            images, labels = _filter_classes(data_dir, images, labels, samples_type)

            if labels.size(0) == 0:
                continue

            # Model inference (raw logits for the whole batch)
            outputs = model(images)  # [batch_size, num_classes] logits

            # Convert logits --> ranked predictions with confidences for each image in batch
            sp_pred_batch = batch_logits_to_predictions(outputs, id2label)

            for i, label_idx in enumerate(labels.tolist()):
                gt_name = id2label[label_idx]
                record_evaluation_sample(
                    artifacts,
                    species_gt=gt_name,
                    species_macro_key=gt_name,
                    sp_pred=sp_pred_batch[i],
                    save_prediction_detail=save_predictions,
                )

    print_species_accuracy_report(run_name, artifacts.state)

    if output_dir:
        write_evaluation_artifacts(
            output_dir, artifacts, save_predictions=save_predictions
        )
        print(f"Wrote evaluation artifacts to {output_dir}", flush=True)

    return artifacts


def compare_evaluations(
    baseline_output_dir: str,
    finetuned_output_dir: str,
    output_dir: str,
    overlap_table_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare baseline and fine-tuned per-species evaluation artifacts.

    Produces:
    - per_species_comparison.csv: Comparison of per-species accuracy between baseline and fine-tuned runs.
    - comparison_report.md: Summary of the comparison.
    - ami_traps_only_comparison.csv: Comparison of per-species accuracy between baseline and fine-tuned runs for AMI-Traps-only species.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparison = compare_per_species_reports(
        baseline_csv=Path(baseline_output_dir) / "per_species_accuracy.csv",
        finetuned_csv=Path(finetuned_output_dir) / "per_species_accuracy.csv",
        output_csv=out / "per_species_comparison.csv",
        overlap_table_csv=overlap_table_csv,
    )

    summary_lines = [
        "# Evaluation comparison",
        "",
        f"- Species compared: {len(comparison)}",
        f"- Improved (top-1): {(comparison['top1_change'] == 'improved').sum()}",
        f"- Regressed (top-1): {(comparison['top1_change'] == 'regressed').sum()}",
        f"- Unchanged (top-1): {(comparison['top1_change'] == 'unchanged').sum()}",
    ]
    if "in_overlap_set" in comparison.columns:
        overlap_df = comparison[comparison["in_overlap_set"] == True]  # noqa: E712
        summary_lines.extend(
            [
                "",
                "## Overlap species subset",
                f"- Species: {len(overlap_df)}",
                f"- Improved: {(overlap_df['top1_change'] == 'improved').sum()}",
                f"- Regressed: {(overlap_df['top1_change'] == 'regressed').sum()}",
                f"- Unchanged: {(overlap_df['top1_change'] == 'unchanged').sum()}",
                "",
                "### Improved overlap species",
                *_format_species_change_list(overlap_df, "improved"),
                "",
                "### Regressed overlap species",
                *_format_species_change_list(overlap_df, "regressed"),
            ]
        )

    ami_traps_only = comparison[
        comparison["n_baseline"].notna() & comparison["n_finetuned"].notna()
    ]
    if overlap_table_csv and Path(overlap_table_csv).exists():
        overlap_species = set(pd.read_csv(overlap_table_csv)["species_name"])
        ami_traps_only = comparison[
            comparison["species"].notna()
            & ~comparison["species"].isin(overlap_species)
            & comparison["n_baseline"].notna()
        ]
        summary_lines.extend(
            [
                "",
                "## AMI-Traps-only species (forgetting-risk subset)",
                f"- Species: {len(ami_traps_only)}",
                f"- Regressed: {(ami_traps_only['top1_change'] == 'regressed').sum()}",
            ]
        )
        ami_traps_only.to_csv(out / "ami_traps_only_comparison.csv", index=False)

    # Write the comparison report
    report = "\n".join(summary_lines) + "\n"
    (out / "comparison_report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    return comparison


if __name__ == "__main__":
    evaluate_model(
        data_dir="~/vanessa/data/fine_tuning_data",
        model_file="~/vanessa/data/models/moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth",
        model_type="resnet50",
        num_classes=3107,
        test_webdataset="~/vanessa/data/fine_tuning_data/webdataset/test/test-{000000..000003}.tar",
        category_map_json="~/vanessa/data/models/quebec-vermont_moth-category-map_19Jan2023.json",
        samples_type="all",
    )
