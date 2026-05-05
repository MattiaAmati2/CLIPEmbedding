#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide the path to the .pt files to run."
    echo "Usage: ./run_experiment.sh [path/to/pt_file.py]"
    exit 1
fi

SOURCES_PREFIX=$1

TRAIN="sources/${SOURCES_PREFIX}_train_embeddings.pt"
VALIDATION="sources/${SOURCES_PREFIX}_validation_embeddings.pt"

echo "Starting Experiments for: $SOURCES_PREFIX"

for shots in 8 16 32 64
do
    python -m classifiers.mahalanobis_distance_ncm --train_filename "$TRAIN" --test_filename "$VALIDATION" --shot_number "$shots"
    python -m classifiers.bayesian_estimation_experiment --train_filename "$TRAIN" --test_filename "$VALIDATION" --shot_number "$shots"
done

echo "======================================"
echo "All experiments for $SOURCES_PREFIX finished!"