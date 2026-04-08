import argparse
import random
import torch
from sklearn.metrics import accuracy_score, f1_score

from data_collection import save_results


def get_class_means_and_inv_covariance_matrices(train_file, shot_number):
    image_embeddings = train_file["image_embeddings"]
    labels = train_file["labels"]

    if isinstance(labels[0], torch.Tensor):
        labels = [lbl.item() for lbl in labels]

    unique_labels = list(set(labels))
    unique_labels.sort()

    class_means = []
    class_matrices = []

    for label in unique_labels:
        valid_indices = [i for i, current_label in enumerate(labels) if current_label == label]

        selected_indices = random.sample(valid_indices, shot_number)

        selected_embeddings = image_embeddings[selected_indices]

        class_mean = selected_embeddings.mean(dim=0)
        class_means.append(class_mean)

        #regularize the covariance matrix to allow its inversion
        cov_matrix = torch.cov(selected_embeddings.T)
        cov_matrix += 0.00001 * torch.eye(512)

        class_matrices.append(torch.linalg.inv(cov_matrix))

    samples_matrix = torch.stack(class_means)

    return samples_matrix, unique_labels, class_matrices


def mahalanobis_distance(test_examples, class_means, covariance_matrices):

    all_distances = []

    #compute the distance from a single class of all the examples at once
    for class_idx in range(len(class_means)):
        mean = class_means[class_idx]
        inv_cov = covariance_matrices[class_idx]

        distances = torch.sum((test_examples - mean) @ inv_cov * (test_examples - mean), dim=1)
        all_distances.append(distances)


    return torch.stack(all_distances, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filename", required=True, type=str)
    parser.add_argument("--test_filename", required=True, type=str)
    parser.add_argument("--shot_number", required=True, type=int)

    args = parser.parse_args()
    train_file = torch.load(args.train_filename)
    test_file = torch.load(args.test_filename)

    class_names = test_file["class_names"]
    ground_truth_labels = test_file["labels"]
    if not isinstance(ground_truth_labels[0], str):
        ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]

    extractions_number = 8
    accuracies= []
    f1_scores = []

    for i in range(extractions_number):

        class_means, unique_labels, class_matrices = get_class_means_and_inv_covariance_matrices(train_file,
                                                                                                args.shot_number)

        distance_matrix = mahalanobis_distance(test_file["image_embeddings"], class_means, class_matrices)

        predictions = (distance_matrix.argmin(dim=1))

        predictions = [class_names[idx.item()] for idx in predictions]

        accuracies.append(accuracy_score(ground_truth_labels, predictions))
        f1_scores.append(f1_score(ground_truth_labels, predictions, average="macro"))

    save_results("mahalanobis_distance_classification_results.csv", args.shot_number, extractions_number, accuracies, f1_scores)


if __name__ == '__main__':
    main()