import torch
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix

from utils.data_collection import save_report_to_csv, save_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.filename)
    image_embeddings = data["image_embeddings"]
    text_embeddings = data["text_embeddings"]
    ground_truth_labels = data["labels"]
    class_names = data["class_names"]

    similarity_scores = image_embeddings @ text_embeddings.T

    predictions = similarity_scores.argmax(dim=1)

    predictions = [class_names[idx.item()] for idx in predictions]

    if not isinstance(ground_truth_labels[0], str):
        ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]

    report_dict = classification_report(ground_truth_labels, predictions, output_dict=True)
    cm = confusion_matrix(ground_truth_labels, predictions, labels=class_names)

    dataset_prefix = os.path.basename(args.filename).replace("_embeddings.pt", "")
    dataset_prefix = "zero_shot_" + str(dataset_prefix)
    dataset_directory = os.path.basename(args.filename).rsplit("_", 2)[0]
    save_confusion_matrix(cm, dataset_directory, dataset_prefix, class_names, 0)
    #save_report_to_csv(report_dict, f"{dataset_prefix}_zero_shot_report.csv")

if __name__ == "__main__":
    main()
