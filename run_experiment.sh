#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide the path to the .pt files to run."
    echo "Usage: ./run_experiment.sh [path/to/pt_file.py]"
    exit 1
fi

SOURCES_PREFIX=$1

TRAIN="sources/${SOURCES_PREFIX}_train_embeddings.pt"
VALIDATION="sources/${SOURCES_PREFIX}_val_embeddings.pt"

echo "Starting Experiments for: $SOURCES_PREFIX"

for shots in 8 16 32 64
do
    echo "======================================"
    echo "Running extraction for $shots shots..."

    for reg_factor in -8 -7 -6 -5 -4 -3 -2
    do
      echo "======================================"
      echo "Running extraction with regularization factor $reg_factor ..."

      PYTHONPATH=$(pwd) python classifiers/mahalanobis_distance_ncm.py --train_filename "$TRAIN" --test_filename "$VALIDATION" --shot_number "$shots" --regularization_factor "$reg_factor"
    done
done

echo "======================================"
echo "All experiments for $SOURCES_PREFIX finished!"