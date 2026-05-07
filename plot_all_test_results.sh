#!/bin/bash

# Ensure Python can find the utils module
export PYTHONPATH=$(pwd)

# Define the root directory where your results are saved
RESULTS_DIR="results"

echo "=================================================="
echo "STARTING RESULTS PLOTTING PIPELINE"
echo "=================================================="

for agg_file in ${RESULTS_DIR}/*/bayesian_*.csv; do
        [ -e "$agg_file" ] || { echo "No aggregate files found in subdirectories of $RESULTS_DIR!"; exit 1; }
    current_dir=$(dirname "$agg_file")

    filename=$(basename "$agg_file")
    temp=${filename#bayesian_}
    prefix=${temp%_val.csv}

    echo "--------------------------------------------------"
    echo "Generating plots for: $prefix"
    echo "Saving to: $current_dir"
    echo "--------------------------------------------------"

    zero_shot_file="${current_dir}/${prefix}_test_zero_shot_report.csv"
    out_acc="plots/${prefix}_bayesian_accuracy_curve.png"
    out_f1="plots/${prefix}_bayesian_f1_curve.png"

    # 1. Plot Accuracy (mu_acc)
    echo "  [1/2] Plotting Accuracy vs Shot Number..."
    python -m utils.plot_results \
        --csv "$agg_file" \
        --x "evidence_weight" \
        --y "mu_acc" \
        --xlabel "Evidence Weight" \
        --ylabel "$\mu_{acc}$" \
        --group "shot_number" \
        --legend_title "Shots" \
        --zero_shot_file "$zero_shot_file" \
        --output "$out_acc"

    # 2. Plot F1 Score (mu_f1)
    echo "  [2/2] Plotting Macro F1 vs Shot Number..."
    python -m utils.plot_results \
        --csv "$agg_file" \
        --x "evidence_weight" \
        --y "mu_f1" \
        --xlabel "Evidence Weight" \
        --ylabel "$\mu_{F1}$" \
        --group "shot_number" \
        --legend_title "Shots" \
        --zero_shot_file "$zero_shot_file" \
        --output "$out_f1"

done

echo "=================================================="
echo "ALL PLOTS GENERATED SUCCESSFULLY!"
echo "Check the individual subfolders in $RESULTS_DIR for your PNG files."
echo "=================================================="