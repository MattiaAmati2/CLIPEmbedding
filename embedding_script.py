import argparse
from tqdm import tqdm
import datasets
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

def find_supervised_keys(dataset):
    image_label = "image"
    text_label = "label"
    class_label_found = False
    features = datasets.get_dataset_config_info(dataset).features
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

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices = ["openai/clip-vit-base-patch16",
                                                                                   "openai/clip-vit-base-patch32"])
    args = parser.parse_args()

    image_label, text_label = find_supervised_keys(args.dataset)

    model = CLIPModel.from_pretrained(args.model)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    label_translator = datasets.get_dataset_config_info(args.dataset).features[text_label]

    def image_preprocess(raw_batch):
        processed_images = processor(images = raw_batch[image_label], return_tensors="pt", padding=True)
        processed_images[text_label] = raw_batch[text_label]

        return processed_images

    splits = datasets.get_dataset_split_names(args.dataset)
    dataset_splits = {}
    for split in splits:
        dataset_splits[split] = load_dataset(args.dataset, split=split)

    unique_labels = sorted(list(set(dataset_splits["train"][text_label])))

    if isinstance(label_translator, datasets.ClassLabel):
        unique_labels = label_translator.int2str(unique_labels)

    unique_labels = [f"A photo of a {label}" for label in unique_labels]
    processed_labels = processor(text = unique_labels, return_tensors="pt", padding = True).to(device)
    text_embeddings = model.get_text_features(**processed_labels).cpu()

    print("text_embeddings.shape", text_embeddings.shape)

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

        results_dict = {
            "image_embeddings": torch.cat(image_embeddings),
            "text_embeddings" : torch.nn.functional.normalize(text_embeddings, p=2, dim=1),
            "labels": text_labels,
            "class_names" : unique_labels
        }

        results_dict["image_embeddings"] = torch.nn.functional.normalize(results_dict["image_embeddings"], p=2, dim=1)
        torch.save(results_dict, f"{args.model.replace('/', '-')}__{args.dataset.replace('/', '-')}__{key}__embeddings.pt")

if __name__ == "__main__":
    main()
