import argparse
import os
from tqdm import tqdm
import datasets
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

def find_supervised_keys(features):
    image_label = "image"
    text_label = "label"
    class_label_found = False

    for key in features:
        if isinstance(features[key], datasets.ClassLabel):
            text_label = key
            class_label_found = True
        if isinstance(features[key], datasets.Image):
            image_label = key
        if isinstance(features[key], datasets.Value) and not class_label_found and features[key].dtype == "string":
            text_label = key

    return image_label, text_label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices = ["openai/clip-vit-base-patch16",
                                                                                   "openai/clip-vit-base-patch32"])

    parser.add_argument("--label_col", type=str, default=None,
                        help="The specific text/label column to use (e.g. fine_label)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_id", type=str, help="A valid Hugging Face ID")
    group.add_argument("--dataset_path", type=str, help="A local path for imagefolder")

    args = parser.parse_args()

    if args.dataset_path:
        dataset_splits = load_dataset("imagefolder", data_dir=args.dataset_path)
        dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    else:
        dataset_splits = load_dataset(args.dataset_id)
        dataset_name = args.dataset_id

    splits = list(dataset_splits.keys())
    first_split = splits[0]

    features = dataset_splits[first_split].features
    image_label, text_label = find_supervised_keys(features)
    if args.label_col is not None:
        text_label = args.label_col

    model = CLIPModel.from_pretrained(args.model)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    label_feature = features[text_label]

    if isinstance(label_feature, datasets.ClassLabel):
        unique_labels = label_feature.names
    else:
        unique_labels = sorted(list(set(dataset_splits[first_split][text_label])))

    results_dict = {
        "class_names": unique_labels,
    }

    text_prompts = [f"A photo of a {label.replace("_", " ")}" for label in unique_labels]

    processed_labels = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    text_embeddings = model.get_text_features(**processed_labels).cpu()

    def image_preprocess(raw_batch):
        processed_images = processor(images = raw_batch[image_label], return_tensors="pt", padding=True)
        processed_images[text_label] = raw_batch[text_label]

        return processed_images

    for split in splits:
        dataset_splits[split] = dataset_splits[split].with_transform(image_preprocess)

    for key in dataset_splits:
        data_loader = torch.utils.data.DataLoader(dataset_splits[key], batch_size=256, shuffle=False, num_workers=16)

        image_embeddings = []
        text_labels = []
        for batch in tqdm(data_loader, desc = "Computing split: " + key):
            with (torch.no_grad()):
                image_batch = batch["pixel_values"].to(device)
                image_embeddings.append(model.get_image_features(image_batch).cpu())
                text_labels.extend(batch[text_label])

        split_results = results_dict.copy()
        split_results["image_embeddings"] = torch.cat(image_embeddings)
        split_results["text_embeddings"] = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
        split_results["labels"] = text_labels

        split_results["image_embeddings"] = torch.nn.functional.normalize(split_results["image_embeddings"], p=2, dim=1)

        torch.save(split_results, f"{args.model.replace('/', '-')}__{dataset_name.replace('/', '-')}__{key}__embeddings.pt")

if __name__ == "__main__":
    main()
