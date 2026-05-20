#!/bin/bash

export PYTHONPATH=$(pwd)

RESULTS_DIR="results"

echo "=================================================="
echo "STARTING RESULTS PLOTTING PIPELINE"
echo "=================================================="

for agg_file in ${RESULTS_DIR}/*/mahalanobis_ncm_*_results.csv; do
        [ -e "$agg_file" ] || { echo "No aggregate files found in subdirectories of $RESULTS_DIR!"; exit 1; }
    current_dir=$(dirname "$agg_file")

    filename=$(basename "$agg_file")
    prefix=${filename%_val_results.csv}
    prefix=${prefix#mahalanobis_ncm_}

    echo "--------------------------------------------------"
    echo "Generating plots for: $prefix"
    echo "--------------------------------------------------"

    zero_shot_file="${current_dir}/${prefix}_test_zero_shot_report.csv"
    out_f1="plots/${prefix}_mahalanobis_validation.pdf"

    # 2. Plot F1 Score (mu_f1)
    python -m utils.plot_results \
        --csv "$agg_file" \
        --x "shot_number" \
        --y "mu_f1" \
        --xlabel "Shots" \
        --ylabel "$\mu_{F1}$" \
        --group "regularization_factor" \
        --legend_title "Regularization Factor $ (10^{\lambda})$" \
        --zero_shot_file "$zero_shot_file" \
        --output "$out_f1"
done

echo "=================================================="
echo "ALL PLOTS GENERATED SUCCESSFULLY!"
echo "=================================================="