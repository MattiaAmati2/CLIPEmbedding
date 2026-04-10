#!/bin/bash

TRAIN="sources/openai-clip-vit-base-patch32_oxford-pets_train_embeddings.pt"
VALIDATION="sources/openai-clip-vit-base-patch32_oxford-pets_validation_embeddings.pt"

echo "Starting Euclidean NCM Experiments..."

for shots in 8 16 32 64 128
do
    echo "======================================"
    echo "Running extraction for $shots shots..."
    python ncm_few_shot_classification.py --train_filename $TRAIN --test_filename $VALIDATION --shot_number $shots
done

echo "All experiments finished!"