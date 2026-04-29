import torch
import os
import pandas as pd
import argparse

from sklearn.metrics import classification_report


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

    report_df = pd.DataFrame(report_dict).transpose().round(4)
    dataset_prefix = os.path.basename(args.filename).replace("_embeddings.pt", "")
    save_path = f"results/{dataset_prefix}_zero_shot_report.csv"

    os.makedirs("results", exist_ok=True)

    report_df.to_csv(save_path, index=True, index_label="Class_Name")

    print(f"[+] Zero-Shot classification report saved to: {save_path}")


if __name__ == "__main__":
    main()
