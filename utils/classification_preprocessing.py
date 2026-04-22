import torch
import random


def get_class_means(train_file, shot_number):
    image_embeddings = train_file["image_embeddings"]
    labels = train_file["labels"]

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

    return samples_matrix


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

        cov_matrix = torch.cov(selected_embeddings.T)
        cov_matrix += 10 ** (-2) * torch.eye(512)

        class_matrices.append(torch.linalg.inv(cov_matrix))

    samples_matrix = torch.stack(class_means)

    return samples_matrix, class_matrices


def mahalanobis_distance(test_examples, class_means, covariance_matrices):
    all_distances = []

    #compute the distance from a single class of all the examples at once
    for class_idx in range(len(class_means)):
        mean = class_means[class_idx]
        inv_cov = covariance_matrices[class_idx]

        distances = torch.sum((test_examples - mean) @ inv_cov * (test_examples - mean), dim=1)
        all_distances.append(distances)

    return torch.stack(all_distances, dim=1)


@torch.no_grad()
def get_segment_points(start_points, end_points, num_steps=10):

    t = torch.linspace(0, 1, num_steps, device=start_points.device)

    start_3d = start_points.unsqueeze(1)
    end_3d = end_points.unsqueeze(1)
    t_3d = t.view(1, num_steps, 1)

    line_points = torch.lerp(start_3d, end_3d, t_3d)

    return line_points