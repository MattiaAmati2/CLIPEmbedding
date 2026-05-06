import argparse

import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from utils.data_collection import save_results, save_report_to_csv, save_confusion_matrix
from utils.classification_preprocessing import get_class_means_and_inv_covariance_matrices, mahalanobis_distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filename", required=True, type=str)
    parser.add_argument("--test_filename", required=True, type=str)
    parser.add_argument("--shot_number", required=True, type=int)
    parser.add_argument("--regularization_factor", required=False, type=int, default=-2)

    args = parser.parse_args()
    train_file = torch.load(args.train_filename)
    test_file = torch.load(args.test_filename)

    dataset_prefix = os.path.basename(args.test_filename).replace("_embeddings.pt", "")
    dataset_prefix = "mahalanobis_" + str(dataset_prefix)

    class_names = test_file["class_names"]
    ground_truth_labels = test_file["labels"]
    if not isinstance(ground_truth_labels[0], str):
        ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]

    extractions_number = 16
    predictions = []
    accuracies = []
    f1_scores = []
    accumulated_cm = np.zeros((len(class_names), len(class_names)), dtype=int)

    for i in range(extractions_number):
        class_means, class_matrices = get_class_means_and_inv_covariance_matrices(train_file, args.shot_number, args.regularization_factor)

        distance_matrix = mahalanobis_distance(test_file["image_embeddings"], class_means, class_matrices)

        predictions = (distance_matrix.argmin(dim=1))

        predictions = [class_names[idx.item()] for idx in predictions]

        accuracies.append(accuracy_score(ground_truth_labels, predictions))
        f1_scores.append(f1_score(ground_truth_labels, predictions, average="macro"))

        current_cm = confusion_matrix(ground_truth_labels, predictions, labels=class_names)
        accumulated_cm += current_cm


    save_results(f"results/{dataset_prefix}_results.csv", args.shot_number, args.regularization_factor,
                 accuracies, f1_scores)

    save_confusion_matrix(accumulated_cm, dataset_prefix, class_names, args.shot_number)

    #report_dict = classification_report(ground_truth_labels, predictions, output_dict=True)
    #save_report_to_csv(report_dict, f"{dataset_prefix}_mhd_ncm_report.csv")

if __name__ == '__main__':
    main()
