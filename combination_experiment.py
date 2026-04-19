import argparse
import torch
from sklearn.metrics import accuracy_score, f1_score

from classification_preprocessing import get_class_means, get_segment_points
from data_collection import save_results


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
    interpolated_points = 16

    accuracies = []
    f1_scores = []

    for i in range(interpolated_points):
        for j in range(extractions_number):
            samples_matrix  = get_class_means(train_file, args.shot_number)

            points = get_segment_points(text_embeddings, samples_matrix, interpolated_points)

            similarity_scores = test_file["image_embeddings"] @ points[:, i, :].T
            predictions = similarity_scores.argmax(dim=1)
            predictions = [class_names[idx.item()] for idx in predictions]

            accuracies.append(accuracy_score(ground_truth_labels, predictions))
            f1_scores.append(f1_score(ground_truth_labels, predictions, average="macro"))

        save_results("results/interpolation_experiment.csv", args.shot_number, i,
                     accuracies, f1_scores)

if __name__ == '__main__':
    main()