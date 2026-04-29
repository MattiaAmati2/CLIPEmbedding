#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide the embedding prefix."
    echo "Usage: ./run_all_evaluations.sh [MODEL_DATASET_PREFIX]"
    echo "Example: ./run_all_evaluations.sh openai-clip-vit-base-patch32_food101"
    exit 1
fi

PREFIX=$1
TRAIN="sources/${PREFIX}_train_embeddings.pt"
TEST="sources/${PREFIX}_val_embeddings.pt"

export PYTHONPATH=$(pwd)

echo "=================================================="
echo "STARTING EVALUATION SUITE FOR: $PREFIX"
echo "=================================================="

# 1. Zero-Shot Baseline
echo "Running Zero-Shot Classification..."
python -m classifiers.zero_shot_classification --filename "$TEST"
echo "--------------------------------------------------"

# 2. Few-Shot Experiments
for shots in 8 16 32 64
do
    echo "======================================"
    echo "RUNNING $shots-SHOT EXPERIMENTS..."
    echo "======================================"

    echo "  [1/4] Standard NCM..."
    python -m classifiers.ncm_few_shot_classification \
        --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$shots"

    echo "  [2/4] Mahalanobis Distance NCM..."
    python -m classifiers.mahalanobis_distance_ncm \
        --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$shots"

    echo "  [3/4] Fixed Interpolation Experiment..."
    python -m classifiers.interpolation_experiment \
        --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$shots"

    echo "  [4/4] Point Optimization Experiment..."
    python -m classifiers.point_selection_experiment \
        --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$shots"
done

echo "=================================================="
echo "ALL EVALUATIONS COMPLETED FOR $PREFIX!"
echo "Check the /results folder for your CSVs."
echo "=================================================="