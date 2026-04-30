#!/bin/bash

PREFIX=$1
BEST_SINGLE_POINT_ACC=$2
BEST_SINGLE_POINT_F1=$3
BEST_REGULARIZATION_FACTOR=$4
SHOTS=$5
TRAIN="sources/${PREFIX}_train_embeddings.pt"
TEST="sources/${PREFIX}_test_embeddings.pt"
OPTIMAL_POINTS="points/${PREFIX}_val_optimal_points.csv"

export PYTHONPATH=$(pwd)

echo "=================================================="
echo "STARTING TEST EVALUATION SUITE FOR: $PREFIX"
echo "=================================================="

# 1. Zero-Shot Baseline
echo "Running Zero-Shot Classification..."
python -m classifiers.zero_shot_classification --filename "$TEST"
echo "--------------------------------------------------"

# 2. Few-Shot Experiments
echo "======================================"
echo "RUNNING $SHOTS-SHOT EXPERIMENTS..."
echo "======================================"

echo "  [1/3] Standard NCM..."
python -m classifiers.ncm_few_shot_classification \
    --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$SHOTS"

echo "  [2/3] Mahalanobis Distance NCM..."
python -m classifiers.mahalanobis_distance_ncm \
    --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$SHOTS" --regularization_factor "$BEST_REGULARIZATION_FACTOR"

echo "  [3/3] Fixed Interpolation Experiment..."
python -m classifiers.test_split_classifications \
    --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$SHOTS" --fixed_points "$BEST_SINGLE_POINT_ACC" "$BEST_SINGLE_POINT_F1"

#Not running this, it has already shown that this approach is doomed to fail
#echo "  [4/4] Point Optimization Experiment..."
#python -m classifiers.test_split_classifications \
#   --train_filename "$TRAIN" --test_filename "$TEST" --shot_number "$SHOTS" --optimal_points_csv "$OPTIMAL_POINTS"


echo "=================================================="
echo "ALL EVALUATIONS COMPLETED FOR $PREFIX!"
echo "Check the /results folder for your CSVs."
echo "=================================================="