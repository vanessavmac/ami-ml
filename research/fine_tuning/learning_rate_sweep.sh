#!/usr/bin/env bash
# LR + unfreeze-depth sweep for Atlantic Forestry fine-tuning.
# Sweeps head-only (fc), fc+layer4, and fc+layer3+layer4 with discriminative LR
# (backbone at 0.1x head LR when backbone stages are unfrozen).
# 100-epoch ceiling with early stopping 15 and 5-epoch warmup. Head LR grid
# includes 1e-3 (strong for head-only) down through 5e-5 (conservative for
# deeper unfreeze). W&B run names use suffix _100ep so reporting can target
# this sweep separately.

ATLANTIC_DATA_DIR="~/vanessa/data/fine_tuning_data_atlantic"
AMI_TRAPS_TEST="~/vanessa/data/fine_tuning_data/webdataset/test/test-{000000..000003}.tar"
QUEBEC_WEIGHTS="~/vanessa/data/models/moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth"
TRAIN_SHARDS="${ATLANTIC_DATA_DIR}/webdataset/train/train-{000000..000022}.tar"
VAL_SHARD="${ATLANTIC_DATA_DIR}/webdataset/val/val-{000000..000004}.tar"

TOTAL_EPOCHS=100
EARLY_STOPPING=15
SWEEP_TAG="_100ep"
BACKBONE_LR_SCALE=0.1

for LR in 1e-3 5e-4 3e-4 1e-4 5e-5; do
  for UNFREEZE in "" "layer4" "layer3,layer4"; do
    if [ -z "$UNFREEZE" ]; then
      UNFREEZE_TAG="fc"
    else
      UNFREEZE_TAG="${UNFREEZE//,/-}"
    fi
    ami-classification train-model \
      --model_type "resnet50" \
      --num_classes 3107 \
      --existing_weights "$QUEBEC_WEIGHTS" \
      --image_input_size 128 \
      --preprocess_mode "torch" \
      --total_epochs "$TOTAL_EPOCHS" \
      --early_stopping "$EARLY_STOPPING" \
      --warmup_epochs 5 \
      --train_webdataset "$TRAIN_SHARDS" \
      --val_webdataset "$VAL_SHARD" \
      --test_webdataset "$AMI_TRAPS_TEST" \
      --batch_size 16 \
      --learning_rate "$LR" \
      --learning_rate_scheduler "cosine" \
      --mixed_resolution_data_aug True \
      --freeze_backbone True \
      --unfreeze_backbone_layers "$UNFREEZE" \
      --backbone_lr_scale "$BACKBONE_LR_SCALE" \
      --model_save_directory "${ATLANTIC_DATA_DIR}/checkpoints" \
      --wandb_entity "moth-ai" \
      --wandb_project "atlantic-forestry" \
      --wandb_run_name "atlantic-forestry_lr${LR}_${UNFREEZE_TAG}${SWEEP_TAG}"
  done
done
