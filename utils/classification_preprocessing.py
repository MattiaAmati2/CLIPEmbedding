import numpy as np
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


def get_class_means_and_inv_covariance_matrices(train_file, shot_number, regularization_factor):
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
        cov_matrix += (10 ** regularization_factor) * torch.eye(512)

        class_matrices.append(torch.linalg.inv(cov_matrix))

    samples_matrix = torch.stack(class_means)

    return samples_matrix, class_matrices


def mahalanobis_distance(test_examples, class_means, inverse_covariance_matrices):
    all_distances = []

    #compute the distance from a single class of all the examples at once
    for class_idx in range(len(class_means)):
        mean = class_means[class_idx]
        inv_cov = inverse_covariance_matrices[class_idx]

        distances = torch.sum((test_examples - mean) @ inv_cov * (test_examples - mean), dim=1)
        all_distances.append(distances)

    return torch.stack(all_distances, dim=1)


def ensure_tensor(x):
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, list):
        if len(x) == 1 and isinstance(x[0], torch.Tensor):
            return x[0]
        elif len(x) > 0 and isinstance(x[0], torch.Tensor):
            return torch.stack(x)
    return torch.as_tensor(x, dtype=torch.float32)


def update_posterior(mu_prior, inv_cov_prior, class_means, inv_cov_evidence, shot_number):
    mu_prior = ensure_tensor(mu_prior)
    inv_cov_prior = ensure_tensor(inv_cov_prior)
    inv_cov_evidence = ensure_tensor(inv_cov_evidence)
    class_means = ensure_tensor(class_means)

    mu_prior = mu_prior.unsqueeze(1)
    class_means = class_means.unsqueeze(1)

    inv_cov_n = inv_cov_prior + (shot_number * inv_cov_evidence)
    cov_n = torch.linalg.inv(inv_cov_n)

    term1 = mu_prior @ inv_cov_prior
    term2 = shot_number * (class_means @ inv_cov_evidence)
    sum_terms = term1 + term2

    mu_n= (sum_terms @ cov_n).squeeze(1)

    return mu_n, inv_cov_n

@torch.no_grad()
def get_segment_points(start_points, end_points, num_steps=10):

    t = torch.linspace(0, 1, num_steps, device=start_points.device)

    start_3d = start_points.unsqueeze(1)
    end_3d = end_points.unsqueeze(1)
    t_3d = t.view(1, num_steps, 1)

    line_points = torch.lerp(start_3d, end_3d, t_3d)

    return line_points