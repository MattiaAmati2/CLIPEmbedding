#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide the embedding prefix."
    echo "Usage: ./plot_all_results.sh [MODEL_DATASET_PREFIX]"
    echo "Example: ./plot_all_results.sh openai-clip-vit-base-patch16_FGVC_Aircraft"
    exit 1
fi

PREFIX=$1
export PYTHONPATH=$(pwd)

echo "=================================================="
echo "PLOTTING ALL RESULTS FOR DATASET $PREFIX"
echo "=================================================="

echo "INTERPOLATION EXPERIMENT PLOTS..."

python -m utils.plot_results --csv "results/interpolation_experiment_${PREFIX}_complete.csv" --x point --y mu_acc --group shot_number --zero_shot_file "results/${PREFIX}_val_zero_shot_report.csv" --output "plots/interpolation_experiment_${PREFIX}_accuracy_plot.png"
python -m utils.plot_results --csv "results/interpolation_experiment_${PREFIX}_complete.csv" --x point --y mu_f1 --group shot_number --zero_shot_file "results/${PREFIX}_val_zero_shot_report.csv" --output "plots/interpolation_experiment_${PREFIX}_f1_plot.png"

echo "MAHALANOBIS NCM EXPERIMENT PLOTS..."

python -m utils.plot_results --csv "results/mahalanobis_ncm_${PREFIX}_val_results.csv" --x shot_number --y mu_acc --group regularization_factor --zero_shot_file "results/${PREFIX}_val_zero_shot_report.csv" --output "plots/mahalanobis_ncm_${PREFIX}_accuracy_plot.png"
python -m utils.plot_results --csv "results/mahalanobis_ncm_${PREFIX}_val_results.csv" --x shot_number --y mu_f1 --group regularization_factor --zero_shot_file "results/${PREFIX}_val_zero_shot_report.csv" --output "plots/mahalanobis_ncm_${PREFIX}_f1_plot.png"

echo "FEW-SHOT AGGREGATE PLOTS..."

python -m utils.plot_results --csv "results/few_shots_aggregate_${PREFIX}_results.csv" --x shot_number --y mu_acc --group distance --zero_shot_file "results/${PREFIX}_val_zero_shot_report.csv" --output "plots/few_shots_aggregate_${PREFIX}_accuracy_plot.png"
python -m utils.plot_results --csv "results/few_shots_aggregate_${PREFIX}_results.csv" --x shot_number --y mu_f1 --group distance --zero_shot_file "results/${PREFIX}_val_zero_shot_report.csv" --output "plots/few_shots_aggregate_${PREFIX}_f1_plot.png"




