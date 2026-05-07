import argparse

import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, f1_score

from utils.classification_preprocessing import get_class_means, get_segment_points
from utils.data_collection import save_results


def evaluate_configuration(test_embeddings, ground_truth_labels, class_names, optimal_anchors):
    similarity_scores = test_embeddings @ optimal_anchors.T
    predictions = similarity_scores.argmax(dim=1)
    predictions = [class_names[idx.item()] for idx in predictions]

    acc = accuracy_score(ground_truth_labels, predictions)
    f1 = f1_score(ground_truth_labels, predictions, average="macro")
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filename", required=True, type=str)
    parser.add_argument("--test_filename", required=True, type=str)
    parser.add_argument("--shot_number", required=True, type=int)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixed_points", nargs="+", type=int)
    group.add_argument("--optimal_points_csv", type=str, help="Path to your val_optimal_points.csv")

    args = parser.parse_args()
    train_file = torch.load(args.train_filename)
    test_file = torch.load(args.test_filename)

    dataset_prefix = os.path.basename(args.test_filename).replace("_embeddings.pt", "")

    class_names = test_file["class_names"]
    ground_truth_labels = test_file["labels"]
    if not isinstance(ground_truth_labels[0], str):
        ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]

    text_embeddings = train_file["text_embeddings"]
    extractions_number = 16
    interpolated_points = 128

    configurations = {}

    if args.fixed_points is not None:

        optimized_acc_point = args.fixed_points[0]
        optimized_f1_point = args.fixed_points[1]

        configurations[f"Optimized_Acc_Fixed_Step_{optimized_acc_point}"] = {
            "steps": [optimized_acc_point] * len(class_names),
            "is_single": True,
            "accs": [],
            "f1s": []
        }

        configurations[f"Optimized_F1_Fixed_Step_{optimized_f1_point}"] = {
            "steps": [optimized_f1_point] * len(class_names),
            "is_single": True,
            "accs": [],
            "f1s": []
        }

    else:
        df = pd.read_csv(args.optimal_points_csv)
        f1_col = f"{args.shot_number}_shots_f1_step"
        recall_col = f"{args.shot_number}_shots_recall_step"
        f1_mapping = dict(zip(df['Class_Name'], df[f1_col]))
        recall_mapping = dict(zip(df['Class_Name'], df[recall_col]))

        # 1. Configuration A: Optimized for Balanced F1
        configurations["Optimized_F1"] = {
            "steps": [int(f1_mapping[c]) for c in class_names],
            "is_single": False,
            "accs": [],
            "f1s": []
        }

        # 2. Configuration B: Optimized for Recall
        configurations["Optimized_Recall"] = {
            "steps": [int(recall_mapping[c]) for c in class_names],
            "is_single": False,
            "accs": [],
            "f1s": []
        }

    for j in range(extractions_number):
        samples_matrix = get_class_means(train_file, args.shot_number)
        points = get_segment_points(text_embeddings, samples_matrix, interpolated_points)

        for config_name, config_data in configurations.items():

            if config_data["is_single"]:
                optimal_anchors = points[:, config_data["steps"][0], :]
            else:
                optimal_anchors = torch.stack([
                    points[c, config_data["steps"][c], :] for c in range(len(class_names))
                ])

            acc, f1 = evaluate_configuration(
                test_file["image_embeddings"],
                ground_truth_labels,
                class_names,
                optimal_anchors
            )

            config_data["accs"].append(acc)
            config_data["f1s"].append(f1)

    for config_name, config_data in configurations.items():
        save_results(
            filename=f"results/{dataset_prefix}_interpolation_evaluation.csv",
            shot_number=args.shot_number,
            extra_metadata=config_name,  # This will output as "Optimized_F1" or "Optimized_Recall" in the CSV
            accuracies=config_data["accs"],
            f1_scores=config_data["f1s"]
        )
        print(f"Evaluation saved for: {config_name} on {args.shot_number} shots")

if __name__ == '__main__':
    main()