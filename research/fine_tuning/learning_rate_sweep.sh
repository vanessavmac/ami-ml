#!/usr/bin/env bash
# LR sweep for Atlantic Forestry fine-tuning.
# Best checkpoints from the 10-epoch sweep landed at epochs 8–9, so this sweep allows
# more training time (30 epochs) with proportionally higher early-stopping patience (8).
# W&B run names use suffix _30ep so reporting can target this sweep separately.

ATLANTIC_DATA_DIR="~/data/fine_tuning_data_atlantic"
AMI_TRAPS_TEST="~/data/fine_tuning_data/webdataset/test/test-{000000..000003}.tar"
QUEBEC_WEIGHTS="~/data/models/moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth"
TRAIN_SHARDS="~/data/fine_tuning_data_atlantic/webdataset/train/train-000000.tar"
VAL_SHARD="~/data/fine_tuning_data_atlantic/webdataset/val/val-000000.tar"

TOTAL_EPOCHS=30
EARLY_STOPPING=8
SWEEP_TAG="_30ep"

for LR in 1e-3 5e-4 3e-4 1e-4; do
  ami-classification train-model \
    --model_type "resnet50" \
    --num_classes 3107 \
    --existing_weights "$QUEBEC_WEIGHTS" \
    --image_input_size 128 \
    --preprocess_mode "torch" \
    --total_epochs "$TOTAL_EPOCHS" \
    --early_stopping "$EARLY_STOPPING" \
    --warmup_epochs 2 \
    --train_webdataset "$TRAIN_SHARDS" \
    --val_webdataset "$VAL_SHARD" \
    --test_webdataset "$AMI_TRAPS_TEST" \
    --batch_size 16 \
    --learning_rate "$LR" \
    --learning_rate_scheduler "cosine" \
    --mixed_resolution_data_aug True \
    --freeze_backbone True \
    --model_save_directory "${ATLANTIC_DATA_DIR}/checkpoints" \
    --wandb_entity "vanessavaleriemac-mila" \
    --wandb_project "atlantic-forestry" \
    --wandb_run_name "atlantic-forestry_lr${LR}${SWEEP_TAG}"
done
