#!/bin/bash

MODELS=(
    "openai/clip-vit-base-patch16"
)

DATASETS=(
    "food101"
    "Alanox/stanford-dogs"
    "Donghyun99/FGVC-Aircraft"
)

echo "Starting Massive Extraction Pipeline..."

for model in "${MODELS[@]}"
do
    echo "=================================================="
    echo "INITIALIZING MODEL: $model"
    echo "=================================================="

    for dataset in "${DATASETS[@]}"
    do
        echo "Processing dataset: $dataset"

        python -m embedding_script \
            --model "$model" \
            --dataset_id "$dataset"

        echo "Finished $dataset!"
        echo "--------------------------------------------------"
    done
done

echo "ALL EMBEDDINGS EXTRACTED SUCCESSFULLY!"