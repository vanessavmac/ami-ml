#!/usr/bin/env python
# coding: utf-8

"""Utility functions"""

import random
import tarfile
import typing as tp

import braceexpand
import numpy as np
import timm
import torch
from timm.scheduler import CosineLRScheduler

from src.classification.constants import (
    AVAILABLE_MODELS,
    COSINE_LR_SCHEDULER,
    CROSS_ENTROPY_LOSS,
    VIT_B16_128,
    WEIGHTED_ORDER_AND_BINARY_LOSS,
)
from src.classification.custom_loss_functions import (
    WeightedOrderAndBinaryCrossEntropyLoss,
)


def set_random_seeds(random_seed: int) -> None:
    """Set random seeds for reproducibility"""

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


def get_optimizer(
    optimizer_type: str,
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
    backbone_lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    """Optimizer definitions.

    When ``backbone_lr_scale != 1.0`` (or any backbone params are trainable),
    builds two param groups: the classification head (``fc``) at
    ``learning_rate`` and remaining trainable backbone params at
    ``learning_rate * backbone_lr_scale``. Head-only training falls back to a
    single param group.
    """

    base_model = _unwrap_model(model)
    head_params = []
    backbone_params = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc.") or name == "fc":
            head_params.append(param)
        else:
            backbone_params.append(param)

    if backbone_params and backbone_lr_scale != 1.0:
        param_groups = [
            {"params": head_params, "lr": learning_rate, "weight_decay": weight_decay},
            {
                "params": backbone_params,
                "lr": learning_rate * backbone_lr_scale,
                "weight_decay": weight_decay,
            },
        ]
        print(
            f"Discriminative LR: head={learning_rate}, "
            f"backbone={learning_rate * backbone_lr_scale} "
            f"(scale={backbone_lr_scale}).",
            flush=True,
        )
    else:
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]

    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(param_groups, momentum=momentum)
    else:
        raise RuntimeError(f"{optimizer_type} optimizer is not implemented.")


def get_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler_type: str,
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> tp.Any:
    """Learning rate scheduler definitions"""

    total_steps = int(total_epochs * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    if lr_scheduler_type == COSINE_LR_SCHEDULER:
        return CosineLRScheduler(
            optimizer,
            t_initial=(total_steps - warmup_steps),
            warmup_t=warmup_steps,
            warmup_prefix=True,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        raise RuntimeError(
            f"{lr_scheduler_type} learning rate scheduler is not implemented."
        )


def get_loss_function(
    loss_function_name: str, label_smoothing: float = 0.0, weight_on_order: float = 0.5
) -> torch.nn.Module:
    """Loss function definitions"""

    if loss_function_name == CROSS_ENTROPY_LOSS:
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_function_name == WEIGHTED_ORDER_AND_BINARY_LOSS:
        return WeightedOrderAndBinaryCrossEntropyLoss(weight_on_order=weight_on_order)
    else:
        raise RuntimeError(f"{loss_function_name} loss is not implemented.")


def _count_files_from_tar(tar_filename: str, ext="jpg") -> int:
    """Count the number of images in a single tar archive"""

    tar = tarfile.open(tar_filename)
    files = [f for f in tar.getmembers() if f.name.endswith(ext)]
    count_files = len(files)
    tar.close()
    return count_files


def get_webdataset_length(sharedurl: str) -> int:
    """Get the total number of images in all webdataset files for a given dataset"""

    tar_filenames = list(braceexpand.braceexpand(sharedurl))
    counts = [_count_files_from_tar(tar_f) for tar_f in tar_filenames]
    return int(sum(counts))


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the inner module when wrapped in DataParallel.
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def freeze_backbone(
    model: torch.nn.Module,
    unfreeze_layers: tp.Optional[list[str]] = None,
) -> int:
    """Freeze all parameters except the classification head and optional stages.

    Always keeps ``fc`` trainable. When ``unfreeze_layers`` is provided (e.g.
    ``["layer4"]`` or ``["layer3", "layer4"]``), those named modules are also
    unfrozen. Default ``None`` preserves head-only fine-tuning.

    Returns the number of trainable parameters.
    """
    base_model = _unwrap_model(model)
    if not hasattr(base_model, "fc"):
        raise RuntimeError(
            "freeze_backbone requires a model with an fc classification head."
        )

    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.fc.parameters():
        param.requires_grad = True

    unfrozen = list(unfreeze_layers) if unfreeze_layers else []
    for layer_name in unfrozen:
        if not hasattr(base_model, layer_name):
            raise RuntimeError(
                f"Cannot unfreeze '{layer_name}': model has no such attribute. "
                f"Valid ResNet stages are typically layer1–layer4."
            )
        for param in getattr(base_model, layer_name).parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stages = ", ".join(["fc"] + unfrozen)
    print(
        f"Frozen backbone (trainable: {stages}): "
        f"{trainable:,} trainable / {total:,} total parameters.",
        flush=True,
    )
    return trainable


def build_model(
    device: str,
    model_type: str,
    num_classes: int,
    existing_weights: tp.Optional[str],
    pretrained: bool = True,
    checkpoint: bool = False,
) -> torch.nn.Module:
    """Model builder

    Args:
        device (str): Device to use for the model.
        model_type (str): Type of the model.
        num_classes (int): Number of classes for classification.
        existing_weights (str, optional): Path to existing weights. Defaults to None.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        checkpoint (bool, optional): Whether to load checkpoint. Defaults to False.
    Returns:
        torch.nn.Module: The model.

    """

    if model_type not in AVAILABLE_MODELS:
        raise RuntimeError(f"Model {model_type} not implemented")

    model_arguments = {"pretrained": pretrained, "num_classes": num_classes}
    if model_type == VIT_B16_128:
        # There is no off-the-shelf ViT model for 128x128 image size,
        # so we use 224x224 model with a custom input image size
        model_type = "vit_base_patch16_224_in21k"
        model_arguments["img_size"] = 128

    model = timm.create_model(model_type, **model_arguments)

    # If available, load existing weights
    if existing_weights:
        print("Loading existing model weights.")
        state_dict = torch.load(existing_weights, map_location=torch.device(device))
        if checkpoint:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict, strict=False)

    # Make use of multiple GPUs, if available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    return model
