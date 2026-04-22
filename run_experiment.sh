#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide a Python file to run."
    echo "Usage: ./run_experiment.sh [path/to/python_file.py]"
    exit 1
fi

PYTHON_SCRIPT=$1

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: File '$PYTHON_SCRIPT' not found!"
    exit 1
fi

TRAIN="sources/openai-clip-vit-base-patch32_oxford-pets_train_embeddings.pt"
VALIDATION="sources/openai-clip-vit-base-patch32_oxford-pets_validation_embeddings.pt"

echo "Starting Experiments for: $PYTHON_SCRIPT"

for shots in 8 16 32 64 128
do
    echo "======================================"
    echo "Running extraction for $shots shots..."

    python "$PYTHON_SCRIPT" --train_filename "$TRAIN" --test_filename "$VALIDATION" --shot_number "$shots"
done

echo "======================================"
echo "All experiments for $PYTHON_SCRIPT finished!"