"""Shared taxonomic micro/macro accuracy helpers for AMI-Traps evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.nn.functional import softmax


def check_prediction(
    gt_label: str, pred_label: List[List[str | float]]
) -> Tuple[int, int]:
    """Check for top1 and top5 prediction.

    Returns 0, 1 for incorrect and correct respectively.
    """
    top1, top5 = 0, 0
    top5_pred = pred_label[:5]
    top5_labels = [item[0] for item in top5_pred]
    if gt_label in top5_labels[:1]:
        top1 = 1
    if gt_label in top5_labels[:5]:
        top5 = 1
    return top1, top5


def get_higher_taxon_pred(
    sp_pred: List[List[str | float]], gbif_taxonomy_hierarchy: dict
) -> Tuple[List[List[str | float]], List[List[str | float]]]:
    """Roll up model species predictions at genus and family level."""
    genus_pred, family_pred = {}, {}

    for prediction in sp_pred:
        sp_key, conf = prediction[0], round(float(prediction[1]), 3)
        genus = gbif_taxonomy_hierarchy[sp_key][0]
        family = gbif_taxonomy_hierarchy[sp_key][1]

        if genus not in genus_pred:
            genus_pred[genus] = conf
        else:
            genus_pred[genus] += conf

        if family not in family_pred:
            family_pred[family] = conf
        else:
            family_pred[family] += conf

    genus_pred_sorted = [
        [taxa, round(conf, 3)]
        for taxa, conf in sorted(
            list(genus_pred.items()), key=lambda item: item[1], reverse=True
        )
    ]
    family_pred_sorted = [
        [taxa, round(conf, 3)]
        for taxa, conf in sorted(
            list(family_pred.items()), key=lambda item: item[1], reverse=True
        )
    ]

    return genus_pred_sorted, family_pred_sorted


def get_higher_taxon_gt(label: str, rank: str, taxonomy_map: pd.DataFrame):
    """Get higher taxon for a ground truth label."""
    if rank == "SPECIES":
        try:
            query = taxonomy_map.loc[taxonomy_map["name"] == label]
            return query["GENUS"].values[0], query["FAMILY"].values[0]
        except IndexError:
            print(f"{label} of rank {rank} not found in the taxonomy database.")
    else:
        try:
            query = taxonomy_map.loc[taxonomy_map["name"] == label]
            return query["FAMILY"].values[0]
        except IndexError:
            print(f"{label} of rank {rank} not found in the taxonomy database.")


def update_taxa_accuracy(
    macro_acc: dict, top1: int, top5: int, gt_label: str, gt_rank: str
) -> dict:
    """Update accuracy data for every class at every taxonomic level."""
    if gt_label not in macro_acc[gt_rank]:
        macro_acc[gt_rank][gt_label] = [0, 0, 0]

    macro_acc[gt_rank][gt_label][0] += top1
    macro_acc[gt_rank][gt_label][1] += top5
    macro_acc[gt_rank][gt_label][2] += 1

    return macro_acc


def calculate_macro_accuracy(macro_acc_taxa: dict) -> Tuple[float, float]:
    """Calculate macro accuracy at the given taxonomic level."""
    top1_acc, top5_acc = [], []

    for taxa in macro_acc_taxa.keys():
        top1_acc.append(macro_acc_taxa[taxa][0] / macro_acc_taxa[taxa][2])
        top5_acc.append(macro_acc_taxa[taxa][1] / macro_acc_taxa[taxa][2])

    macro_top1_acc = round(sum(top1_acc) / len(top1_acc) * 100, 2)
    macro_top5_acc = round(sum(top5_acc) / len(top5_acc) * 100, 2)

    return macro_top1_acc, macro_top5_acc


def load_id2label(category_map_json: str) -> Dict[int, str]:
    """Load category map JSON (name -> idx) and return idx -> name."""
    with open(category_map_json, "r", encoding="utf-8") as f:
        categories_map = json.load(f)
    return {categories_map[categ]: categ for categ in categories_map}


def batch_logits_to_predictions(
    outputs: torch.Tensor, id2label: Dict[int, str]
) -> List[List[List[str | float]]]:
    """Convert batch logits to per-sample ranked predictions with confidences."""
    probs = softmax(outputs, dim=1)
    values, indices = torch.topk(probs, probs.size(1), dim=1)
    values = values.cpu().numpy()
    indices = indices.cpu().numpy()

    batch_preds = []
    for batch_idx in range(indices.shape[0]):
        sample_preds = []
        for rank in range(indices.shape[1]):
            idx = int(indices[batch_idx, rank])
            label = id2label[idx]
            sample_preds.append([label, float(values[batch_idx, rank])])
        batch_preds.append(sample_preds)

    return batch_preds


@dataclass
class TaxonomicAccuracyState:
    """Accumulator for species-level micro and macro accuracy."""

    sp_top1: int = 0
    sp_count: int = 0
    macro_acc: dict = field(
        default_factory=lambda: {"SPECIES": {}, "GENUS": {}, "FAMILY": {}}
    )


def record_species_metrics(
    state: TaxonomicAccuracyState,
    species_gt: str,
    species_macro_key: str,
    sp_pred: List[List[str | float]],
) -> TaxonomicAccuracyState:
    """Record species-level micro and macro metrics for one sample."""
    top1, top5 = check_prediction(species_gt, sp_pred)
    state.sp_top1 += top1
    state.sp_count += 1
    update_taxa_accuracy(state.macro_acc, top1, top5, species_macro_key, "SPECIES")
    return state


def print_species_accuracy_report(run_name: str, state: TaxonomicAccuracyState) -> None:
    """Print species-level micro and macro top-1 accuracy."""
    if state.sp_count == 0:
        print(f"No samples evaluated for {run_name}.", flush=True)
        return

    micro_sp_top1 = round(state.sp_top1 / state.sp_count * 100, 2)
    macro_sp_top1, _ = calculate_macro_accuracy(state.macro_acc["SPECIES"])

    print(
        f"\nFine-grained classification micro-accuracy (Top1) for {run_name}:"
        f"\nSpecies: {micro_sp_top1}%"
        f"\n"
        f"\nFine-grained classification macro-accuracy (Top1) for {run_name}:"
        f"\nSpecies: {macro_sp_top1}%"
        f"\n",
        flush=True,
    )


########################################################################################
# Metrics to track per-species accuracy
########################################################################################


@dataclass
class PerSpeciesRecord:
    """Per-species accuracy accumulator."""

    n: int = 0
    top1_correct: int = 0
    top5_correct: int = 0


@dataclass
class EvaluationArtifacts:
    """Detailed evaluation outputs for reporting and comparison."""

    run_name: str
    state: TaxonomicAccuracyState
    per_species: dict[str, PerSpeciesRecord] = field(default_factory=dict)
    gt_pred_pairs: list[tuple[str, str]] = field(default_factory=list)
    prediction_details: list[dict[str, str | bool]] = field(default_factory=list)


def record_evaluation_sample(
    artifacts: EvaluationArtifacts,
    species_gt: str,
    species_macro_key: str,
    sp_pred: List[List[str | float]],
    save_prediction_detail: bool = False,
) -> EvaluationArtifacts:
    """Record micro/macro metrics and optional per-species detail for one sample."""
    top1, top5 = check_prediction(species_gt, sp_pred)
    pred_top1 = sp_pred[0][0] if sp_pred else ""

    record_species_metrics(
        artifacts.state,
        species_gt=species_gt,
        species_macro_key=species_macro_key,
        sp_pred=sp_pred,
    )

    if species_gt not in artifacts.per_species:
        artifacts.per_species[species_gt] = PerSpeciesRecord()
    rec = artifacts.per_species[species_gt]
    rec.n += 1
    rec.top1_correct += top1
    rec.top5_correct += top5

    artifacts.gt_pred_pairs.append((species_gt, str(pred_top1)))
    if save_prediction_detail:
        artifacts.prediction_details.append(
            {
                "gt_species": species_gt,
                "pred_species": str(pred_top1),
                "top1_correct": bool(top1),
            }
        )
    return artifacts


def build_per_species_accuracy_df(
    per_species: dict[str, PerSpeciesRecord],
) -> pd.DataFrame:
    """Build a per-species accuracy table."""
    rows = []
    for species, rec in sorted(per_species.items()):
        top1_acc = round(rec.top1_correct / rec.n * 100, 2) if rec.n else 0.0
        top5_acc = round(rec.top5_correct / rec.n * 100, 2) if rec.n else 0.0
        rows.append(
            {
                "species": species,
                "n": rec.n,
                "top1_correct": rec.top1_correct,
                "top1_acc": top1_acc,
                "top5_correct": rec.top5_correct,
                "top5_acc": top5_acc,
            }
        )
    return pd.DataFrame(rows)


def build_confusion_matrix_df(
    gt_pred_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Build a long-format confusion matrix from (gt, pred) pairs."""
    if not gt_pred_pairs:
        return pd.DataFrame(columns=["gt_species", "pred_species", "count"])

    gt_labels = [gt for gt, _ in gt_pred_pairs]
    pred_labels = [pred for _, pred in gt_pred_pairs]
    labels = sorted(set(gt_labels) | set(pred_labels))
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels)

    rows = []
    for gt_idx, gt_species in enumerate(labels):
        for pred_idx, pred_species in enumerate(labels):
            count = int(cm[gt_idx, pred_idx])
            if count > 0:
                rows.append(
                    {
                        "gt_species": gt_species,
                        "pred_species": pred_species,
                        "count": count,
                    }
                )
    return pd.DataFrame(rows)


def build_evaluation_summary(artifacts: EvaluationArtifacts) -> dict:
    """Summarize micro/macro accuracy for JSON export."""
    state = artifacts.state
    if state.sp_count == 0:
        return {
            "run_name": artifacts.run_name,
            "n_samples": 0,
            "micro_top1_acc": None,
            "macro_top1_acc": None,
        }

    micro_top1 = round(state.sp_top1 / state.sp_count * 100, 2)
    macro_top1, _ = calculate_macro_accuracy(state.macro_acc["SPECIES"])
    return {
        "run_name": artifacts.run_name,
        "n_samples": state.sp_count,
        "micro_top1_acc": micro_top1,
        "macro_top1_acc": macro_top1,
    }


def write_evaluation_artifacts(
    output_dir: str | Path,
    artifacts: EvaluationArtifacts,
    save_predictions: bool = False,
) -> Path:
    """Write per-species metrics, confusion matrix, and summary JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_species_df = build_per_species_accuracy_df(artifacts.per_species)
    per_species_df.to_csv(out / "per_species_accuracy.csv", index=False)

    confusion_df = build_confusion_matrix_df(artifacts.gt_pred_pairs)
    confusion_df.to_csv(out / "confusion_matrix_long.csv", index=False)

    if save_predictions and artifacts.prediction_details:
        pd.DataFrame(artifacts.prediction_details).to_csv(
            out / "predictions_detail.csv", index=False
        )

    summary = build_evaluation_summary(artifacts)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out


def compare_per_species_reports(
    baseline_csv: str | Path,
    finetuned_csv: str | Path,
    output_csv: str | Path,
    overlap_table_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Compare per-species accuracy between baseline and fine-tuned runs."""
    baseline = pd.read_csv(baseline_csv)
    finetuned = pd.read_csv(finetuned_csv)

    merged = baseline.merge(
        finetuned,
        on="species",
        how="outer",
        suffixes=("_baseline", "_finetuned"),
    )
    merged["n"] = merged["n_baseline"].fillna(merged["n_finetuned"])
    merged["delta_top1_acc"] = (
        merged["top1_acc_finetuned"] - merged["top1_acc_baseline"]
    )
    merged["delta_top5_acc"] = (
        merged["top5_acc_finetuned"] - merged["top5_acc_baseline"]
    )

    def _change_label(delta: float) -> str:
        if pd.isna(delta):
            return "unknown"
        if delta > 0:
            return "improved"
        if delta < 0:
            return "regressed"
        return "unchanged"

    merged["top1_change"] = merged["delta_top1_acc"].apply(_change_label)

    if overlap_table_csv and Path(overlap_table_csv).exists():
        overlap = pd.read_csv(overlap_table_csv)
        overlap_species = set(overlap["species_name"])
        merged["in_overlap_set"] = merged["species"].isin(overlap_species)
    else:
        merged["in_overlap_set"] = False

    merged.to_csv(output_csv, index=False)
    return merged
