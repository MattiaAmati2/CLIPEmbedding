#!/bin/bash

# Ensure Python can find the utils module
export PYTHONPATH=$(pwd)

# Define the root directory where your results are saved
RESULTS_DIR="results"

echo "=================================================="
echo "STARTING RESULTS PLOTTING PIPELINE"
echo "=================================================="

for agg_file in ${RESULTS_DIR}/*/*_few_shots_aggregate_results.csv; do
        [ -e "$agg_file" ] || { echo "No aggregate files found in subdirectories of $RESULTS_DIR!"; exit 1; }
    current_dir=$(dirname "$agg_file")

    filename=$(basename "$agg_file")
    prefix=${filename%_few_shots_aggregate_results.csv}

    echo "--------------------------------------------------"
    echo "Generating plots for: $prefix"
    echo "Saving to: $current_dir"
    echo "--------------------------------------------------"

    zero_shot_file="${current_dir}/${prefix}_test_zero_shot_report.csv"
    out_acc="${current_dir}/${prefix}_test_accuracy_curve.png"
    out_f1="${current_dir}/${prefix}_test_f1_curve.png"

    # 1. Plot Accuracy (mu_acc)
    echo "  [1/2] Plotting Accuracy vs Shot Number..."
    python -m utils.plot_results \
        --csv "$agg_file" \
        --x "shot_number" \
        --y "mu_acc" \
        --group "distance" \
        --zero_shot_file "$zero_shot_file" \
        --output "$out_acc"

    # 2. Plot F1 Score (mu_f1)
    echo "  [2/2] Plotting Macro F1 vs Shot Number..."
    python -m utils.plot_results \
        --csv "$agg_file" \
        --x "shot_number" \
        --y "mu_f1" \
        --group "distance" \
        --zero_shot_file "$zero_shot_file" \
        --output "$out_f1"

done

echo "=================================================="
echo "ALL PLOTS GENERATED SUCCESSFULLY!"
echo "Check the individual subfolders in $RESULTS_DIR for your PNG files."
echo "=================================================="