#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide the embedding prefix."
    echo "Usage: ./run_all_evaluations.sh [MODEL_DATASET_PREFIX]"
    echo "Example: ./run_all_evaluations.sh openai-clip-vit-base-patch32_food101"
    exit 1
fi

PREFIX=$1
BEST_SINGLE_POINT=$2
BEST_REGULARIZATION_FACTOR=$3
TRAIN="sources/${PREFIX}_train_embeddings.pt"
TEST="sources/${PREFIX}_test_embeddings.pt"
OPTIMAL_POINTS="results/${PREFIX}_val_optimal_points.csv"

export PYTHONPATH=$(pwd)

echo "=================================================="
echo "STARTING TEST EVALUATION SUITE FOR: $PREFIX"
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
        --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$shots" --regularization_factor "$BEST_REGULARIZATION_FACTOR"

    echo "  [3/4] Fixed Interpolation Experiment..."
    python -m classifiers.test_split_classifications \
        --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$shots" --single_point "$BEST_SINGLE_POINT"

    echo "  [4/4] Point Optimization Experiment..."
    python -m classifiers.test_split_classifications \
        --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$shots" --optimal_points_csv "$OPTIMAL_POINTS"

done

echo "=================================================="
echo "ALL EVALUATIONS COMPLETED FOR $PREFIX!"
echo "Check the /results folder for your CSVs."
echo "=================================================="