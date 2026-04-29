import argparse
import torch
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.data_collection import save_results
from utils.classification_preprocessing import get_class_means_and_inv_covariance_matrices, mahalanobis_distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filename", required=True, type=str)
    parser.add_argument("--test_filename", required=True, type=str)
    parser.add_argument("--shot_number", required=True, type=int)

    args = parser.parse_args()
    train_file = torch.load(args.train_filename)
    test_file = torch.load(args.test_filename)

    dataset_prefix = os.path.basename(args.test_filename).replace("_embeddings.pt", "")

    class_names = test_file["class_names"]
    ground_truth_labels = test_file["labels"]
    if not isinstance(ground_truth_labels[0], str):
        ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]

    extractions_number = 16
    predictions = []
    accuracies = []
    f1_scores = []

    for i in range(extractions_number):
        class_means, class_matrices = get_class_means_and_inv_covariance_matrices(train_file, args.shot_number)

        distance_matrix = mahalanobis_distance(test_file["image_embeddings"], class_means, class_matrices)

        predictions = (distance_matrix.argmin(dim=1))

        predictions = [class_names[idx.item()] for idx in predictions]

        accuracies.append(accuracy_score(ground_truth_labels, predictions))
        f1_scores.append(f1_score(ground_truth_labels, predictions, average="macro"))

    save_results(f"results/mahalanobis_ncm_{dataset_prefix}_results.csv", args.shot_number, extractions_number,
                 accuracies, f1_scores)

    print(classification_report(ground_truth_labels, predictions, digits=4))

if __name__ == '__main__':
    main()
