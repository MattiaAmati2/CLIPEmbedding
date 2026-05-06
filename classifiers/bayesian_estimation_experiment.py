import argparse

import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from utils.data_collection import save_results, save_confusion_matrix
from utils.classification_preprocessing import mahalanobis_distance, \
    get_class_means_and_inv_covariance_matrices, update_posterior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filename", required=True, type=str)
    parser.add_argument("--test_filename", required=True, type=str)
    parser.add_argument("--shot_number", required=True, type=int)

    args = parser.parse_args()
    train_file = torch.load(args.train_filename)
    test_file = torch.load(args.test_filename)

    dataset_prefix = os.path.basename(args.test_filename).replace("_embeddings.pt", "")
    dataset_prefix = "bayesian_" + str(dataset_prefix)
    dataset_directory = os.path.basename(args.test_filename).rsplit("_", 2)[0]

    class_names = test_file["class_names"]
    ground_truth_labels = test_file["labels"]
    if not isinstance(ground_truth_labels[0], str):
        ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]


    target_prior_percentage = 0.5
    evidence_lambda = -2

    if target_prior_percentage >= 1.0:
        prior_pseudo_count = 1e9  # Effectively 100% prior
    else:
        prior_pseudo_count = args.shot_number * (target_prior_percentage / (1.0 - target_prior_percentage))

    baseline_shot_precision = 1.0 / (10**evidence_lambda)
    prior_precision_scalar = prior_pseudo_count * baseline_shot_precision
    prior_inv_covariance_matrix = torch.eye(512) * prior_precision_scalar

    prior_means = test_file["text_embeddings"]

    extractions_number = 16
    accuracies = []
    f1_scores = []
    accumulated_cm = np.zeros((len(class_names), len(class_names)), dtype=int)

    with torch.no_grad():
        for i in range(extractions_number):
            mu_obs, inv_cov_obs = get_class_means_and_inv_covariance_matrices(train_file, args.shot_number, evidence_lambda)
            mu_posterior, inv_cov_posterior = update_posterior(prior_means, prior_inv_covariance_matrix, mu_obs, inv_cov_obs, args.shot_number)

            distance_matrix = mahalanobis_distance(test_file["image_embeddings"], mu_posterior, inv_cov_posterior)

            predictions = (distance_matrix.argmin(dim=1))
            predictions = [class_names[idx.item()] for idx in predictions]

            accuracies.append(accuracy_score(ground_truth_labels, predictions))
            f1_scores.append(f1_score(ground_truth_labels, predictions, average="macro"))

            current_cm = confusion_matrix(ground_truth_labels, predictions, labels=class_names)
            accumulated_cm += current_cm

        save_confusion_matrix(accumulated_cm, dataset_directory, dataset_prefix, class_names, args.shot_number)

        save_results(f"results/{dataset_directory}/{dataset_prefix}.csv", args.shot_number,
                     ["Bayesian", prior_precision_scalar, evidence_lambda],
                     accuracies, f1_scores)

if __name__ == '__main__':
    main()