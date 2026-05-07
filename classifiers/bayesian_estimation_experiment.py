import argparse

import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.data_collection import save_results, read_optimal_weight, save_confusion_matrix
from utils.classification_preprocessing import mahalanobis_distance, \
    get_class_means_and_inv_covariance_matrices, update_posterior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filename", required=True, type=str)
    parser.add_argument("--test_filename", required=True, type=str)
    parser.add_argument("--shot_number", required=True, type=int)
    parser.add_argument("--test_evaluation", default=False, type=bool)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = torch.load(args.train_filename)
    test_file = torch.load(args.test_filename)

    dataset_prefix = os.path.basename(args.test_filename).replace("_embeddings.pt", "")
    dataset_prefix = "bayesian_" + str(dataset_prefix)
    dataset_directory = os.path.basename(args.test_filename).rsplit("_", 2)[0]

    class_names = test_file["class_names"]
    ground_truth_labels = test_file["labels"]
    if not isinstance(ground_truth_labels[0], str):
        ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]

    test_file["image_embeddings"] = test_file["image_embeddings"].to(device)

    prior_means = test_file["text_embeddings"]
    extractions_number = 16

    evidence_lambda = -2

    if args.test_evaluation:
        target_evidence_percentages = read_optimal_weight(dataset_directory, args.shot_number)
    else:
        target_evidence_percentages = [0.5, 0.75]

    baseline_shot_precision = 1.0 / (10 ** evidence_lambda)
    results = {
        weight: {
            "accuracies": [],
            "f1_scores": [],
            "cm": np.zeros((len(class_names), len(class_names)), dtype=int)
        }
        for weight in target_evidence_percentages
    }

    with torch.no_grad():
        for i in range(extractions_number):

            # Compute the evidence once for this extraction
            mu_obs, inv_cov_obs = get_class_means_and_inv_covariance_matrices(train_file, args.shot_number,
                                                                              evidence_lambda)

            # Test all text weights against this single extraction
            for ev_pct in target_evidence_percentages:

                if ev_pct >= 1.0:
                    # 100% Evidence means we completely mute the Prior
                    prior_pseudo_count = 0.0
                elif ev_pct <= 0.0:
                    # 0% Evidence means the Prior dominates completely
                    prior_pseudo_count = 1e9
                else:
                    # The flipped algebraic formula: k = n * ((1 - W_evidence) / W_evidence)
                    prior_pseudo_count = args.shot_number * ((1.0 - ev_pct) / ev_pct)

                prior_precision_scalar = prior_pseudo_count * baseline_shot_precision
                prior_inv_covariance_matrix = torch.eye(512) * prior_precision_scalar

                mu_posterior, inv_cov_posterior = update_posterior(
                    prior_means, prior_inv_covariance_matrix, mu_obs, inv_cov_obs, args.shot_number
                )

                mu_posterior = mu_posterior.to(device)
                inv_cov_posterior = inv_cov_posterior.to(device)

                distance_matrix = mahalanobis_distance(test_file["image_embeddings"], mu_posterior, inv_cov_posterior)

                # Get predictions
                predictions = distance_matrix.argmin(dim=1).cpu()
                predictions_names = [class_names[idx.item()] for idx in predictions]

                # Store metrics
                results[ev_pct]["accuracies"].append(accuracy_score(ground_truth_labels, predictions_names))
                results[ev_pct]["f1_scores"].append(
                    f1_score(ground_truth_labels, predictions_names, average="macro"))
                results[ev_pct]["cm"] += confusion_matrix(ground_truth_labels, predictions_names,
                                                              labels=class_names)

                del distance_matrix


        for weight in target_evidence_percentages:
            save_confusion_matrix(results[weight]["cm"], dataset_directory, dataset_prefix, class_names, args.shot_number)

            save_results(f"results/{dataset_directory}/{dataset_prefix}.csv", args.shot_number,
                         weight, results[weight]["accuracies"], results[weight]["f1_scores"])

if __name__ == '__main__':
    main()