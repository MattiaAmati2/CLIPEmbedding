import argparse
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def get_class_means(train_file, shot_number):
    image_embeddings = train_file["image_embeddings"]
    labels = train_file["labels"]

    #turn the labels from monodimensional tensors to integers to avoid extremely long computation time
    if isinstance(labels[0], torch.Tensor):
        labels = [lbl.item() for lbl in labels]

    unique_labels = list(set(labels))
    unique_labels.sort()

    class_means = []

    for label in unique_labels:
        valid_indices = [i for i, current_label in enumerate(labels) if current_label == label]

        selected_indices = random.sample(valid_indices, shot_number)

        selected_embeddings = image_embeddings[selected_indices]

        class_mean = selected_embeddings.mean(dim=0)
        class_means.append(class_mean)

    samples_matrix = torch.stack(class_means)

    samples_matrix = torch.nn.functional.normalize(samples_matrix, p=2, dim=1)

    return samples_matrix, unique_labels



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
    predictions = []

    extractions_number = 5
    accuracies= []
    f1_scores = []


    for i in range(extractions_number):

        samples_matrix, ordered_train_labels = get_class_means(train_file, args.shot_number)
        similarity_scores = test_file["image_embeddings"] @ samples_matrix.T
        predictions = similarity_scores.argmax(dim=1)

        predictions = [class_names[idx.item()] for idx in predictions]

        if not isinstance(ground_truth_labels[0], str):
            ground_truth_labels = [class_names[label.item()] for label in ground_truth_labels]

        accuracies.append(accuracy_score(ground_truth_labels, predictions))
        f1_scores.append(f1_score(ground_truth_labels, predictions, average="macro"))

    mu_acc = np.mean(accuracies)
    var_acc = np.var(accuracies)
    std_acc = np.std(accuracies)

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy:  {mu_acc:.4f} ± {std_acc:.4f} (Variance: {var_acc:.6f})")


    print("\nSample Report (from final trial):")
    print(classification_report(ground_truth_labels, predictions))

if __name__ == '__main__':
    main()