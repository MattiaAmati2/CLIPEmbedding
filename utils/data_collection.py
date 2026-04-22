import csv
import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def save_results(filename, shot_number, extractions_number, accuracies, f1_scores):
    file_exists = os.path.isfile(filename)

    mu_acc = np.mean(accuracies)
    var_acc = np.var(accuracies)
    std_acc = np.std(accuracies)

    mu_f1 = np.mean(f1_scores)
    var_f1 = np.var(f1_scores)
    std_f1 = np.std(f1_scores)

    row_data = [
        shot_number,
        extractions_number,
        round(mu_acc, 4),
        round(var_acc, 6),
        round(std_acc, 4),
        round(mu_f1, 4),
        round(var_f1, 6),
        round(std_f1, 4)
    ]

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "shot_number",
                "",
                "mu_acc",
                "var_acc",
                "std_acc",
                "mu_f1",
                "var_f1",
                "std_f1"
            ])

        writer.writerow(row_data)

    print(f"\nResults successfully appended to {filename}")


def append_columns_to_csv(filename, class_names, new_columns_dict):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame({"Class_Name": class_names})

    for base_name, data in new_columns_dict.items():
        unique_name = base_name
        counter = 1

        while unique_name in df.columns:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        df[unique_name] = data

    df.to_csv(filename, index=False)
    print(f"[+] Successfully updated '{filename}' with new columns.")


def extract_optimal_metrics(metric_dict, exp_name, metric_label):
    final_mean, final_var, final_std = [], [], []

    for step_index, raw_data in metric_dict.items():
        matrix = np.array(raw_data)
        final_mean.append(np.mean(matrix, axis=0))
        final_var.append(np.var(matrix, axis=0))
        final_std.append(np.std(matrix, axis=0))

    smoothed_mean = uniform_filter1d(np.array(final_mean), size=5, axis=0)

    final_var_array = np.array(final_var)
    final_std_array = np.array(final_std)

    points_selected = np.argmax(smoothed_mean, axis=0)
    scores_for_points = np.max(smoothed_mean, axis=0)

    num_classes = final_var_array.shape[1]
    column_indices = np.arange(num_classes)
    variance_per_class = final_var_array[points_selected, column_indices]
    deviation_per_class = final_std_array[points_selected, column_indices]

    points_selected = np.append(points_selected, np.mean(points_selected))
    scores_for_points = np.append(scores_for_points, np.mean(scores_for_points))
    variance_per_class = np.append(variance_per_class, np.mean(variance_per_class))
    deviation_per_class = np.append(deviation_per_class, np.mean(deviation_per_class))

    columns_to_add = {

        f"{exp_name}_avg_{metric_label}": np.round(scores_for_points, 4),
        #f"{exp_name}_{metric_label}_variance": np.round(variance_per_class, 6),
        #f"{exp_name}_{metric_label}_deviation": np.round(deviation_per_class, 6),
        f"{exp_name}_{metric_label}_step": points_selected
    }

    return columns_to_add