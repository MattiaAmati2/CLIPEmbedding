import csv
import os
import numpy as np

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
                "extractions_number",
                "mu_acc",
                "var_acc",
                "std_acc",
                "mu_f1",
                "var_f1",
                "std_f1"
            ])

        writer.writerow(row_data)

    print(f"\nResults successfully appended to {filename}")

    pass