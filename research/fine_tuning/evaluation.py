#!/usr/bin/env python
# coding: utf-8

""" Evaluation of models on AMI Traps test set
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
from typing import Literal, Tuple

import dotenv
import torch

from src.classification.dataloader import build_webdataset_pipeline
from src.classification.taxonomic_metrics import (
    TaxonomicAccuracyState,
    batch_logits_to_predictions,
    load_id2label,
    print_species_accuracy_report,
    record_species_metrics,
)
from src.classification.utils import build_model

dotenv.load_dotenv()


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
) -> None:
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
    """

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device, model_type, num_classes, model_file, checkpoint=False)
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

    state = TaxonomicAccuracyState()

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
            outputs = model(images) # [batch_size, num_classes] logits

            # Convert logits --> ranked predictions with confidences for each image in batch
            sp_pred_batch = batch_logits_to_predictions(outputs, id2label)

            for i, label_idx in enumerate(labels.tolist()):
                gt_name = id2label[label_idx]
                record_species_metrics(
                    state,
                    species_gt=gt_name,
                    species_macro_key=gt_name,
                    sp_pred=sp_pred_batch[i],
                )

    print_species_accuracy_report(run_name, state)


if __name__ == "__main__":
    evaluate_model(
        data_dir="./fine_tuning_data",  # only needed for seen/unseen
        model_file="moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth",
        model_type="resnet50",
        num_classes=3107,
        test_webdataset="./fine_tuning_data/webdataset/test/test-{000000..000003}.tar",  # fix range
        category_map_json="./fine_tuning_data/quebec-vermont_moth-category-map_19Jan2023.json",
        samples_type="all",
    )