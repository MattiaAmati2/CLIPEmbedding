import torch
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

    predicted_indices = similarity_scores.argmax(dim=1)

    predicted_labels = [class_names[idx.item()] for idx in predicted_indices]
    print(classification_report(ground_truth_labels, predicted_labels))



if __name__ == "__main__":
    main()
