#!/bin/bash

export PYTHONPATH=$(pwd)

RESULTS_DIR="results"

echo "=================================================="
echo "STARTING RESULTS PLOTTING PIPELINE"
echo "=================================================="

for agg_file in ${RESULTS_DIR}/*/interpolation_experiment_*_complete.csv; do
        [ -e "$agg_file" ] || { echo "No aggregate files found in subdirectories of $RESULTS_DIR!"; exit 1; }
    current_dir=$(dirname "$agg_file")

    filename=$(basename "$agg_file")
    prefix=${filename%complete.csv}
    prefix=${prefix#interpolation_experiment_}

    echo "--------------------------------------------------"
    echo "Generating plots for: $prefix"
    echo "--------------------------------------------------"

    zero_shot_file="${current_dir}/${prefix}_test_zero_shot_report.csv"
    out_f1="plots/${prefix}_linear_interpolation.pdf"

    # 2. Plot F1 Score (mu_f1)
    python -m utils.plot_results \
        --csv "$agg_file" \
        --x "point" \
        --y "mu_f1" \
        --xlabel "$\alpha$ (Interpolation Coefficient)" \
        --ylabel "$\mu_{F1}$" \
        --group "shot_number" \
        --legend_title "Shots" \
        --zero_shot_file "$zero_shot_file" \
        --output "$out_f1"
done

echo "=================================================="
echo "ALL PLOTS GENERATED SUCCESSFULLY!"
echo "=================================================="