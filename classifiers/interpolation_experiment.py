import argparse
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score

from utils.classification_preprocessing import get_class_means, get_segment_points
from utils.data_collection import save_results


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

    text_embeddings = train_file["text_embeddings"]
    extractions_number = 16
    interpolated_points = 128

    accuracies = defaultdict(list)
    f1_scores = defaultdict(list)

    for j in range(extractions_number):
        samples_matrix = get_class_means(train_file, args.shot_number)
        for i in range(interpolated_points):
            points = get_segment_points(text_embeddings, samples_matrix, interpolated_points)

            similarity_scores = test_file["image_embeddings"] @ points[:, i, :].T
            predictions = similarity_scores.argmax(dim=1)
            predictions = [class_names[idx.item()] for idx in predictions]

            accuracies[i + 1].append(accuracy_score(ground_truth_labels, predictions))
            f1_scores[i + 1].append(f1_score(ground_truth_labels, predictions, average="macro"))

    for key in accuracies.keys():
        save_results(f"results/interpolation_experiment_{args.shot_number}_shots_{interpolated_points}_points.csv", args.shot_number, key, accuracies[key], f1_scores[key])

if __name__ == '__main__':
    main()