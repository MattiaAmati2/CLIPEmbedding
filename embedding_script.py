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

    parser.add_argument("--model", type=str, required=True, choices = ["openai/clip-vit-base-patch16",
                                                                                   "openai/clip-vit-base-patch32"])
    parser.add_argument("--dataset", type=str, required=True)

    args : argparse.Namespace = parser.parse_args()

    image_label, text_label = find_supervised_keys(args.dataset)

    model = CLIPModel.from_pretrained(args.model)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    label_translator = datasets.get_dataset_config_info(args.dataset).features[text_label]

    def image_preprocess(raw_batch):
        if isinstance(label_translator, datasets.ClassLabel):
            raw_batch[text_label] = label_translator.int2str(raw_batch[text_label])

        raw_batch[text_label] = ["A photo of a " + item for item in raw_batch[text_label]]

        return processor(text=raw_batch[text_label], images=raw_batch[image_label],
                         return_tensors="pt", padding=True)
    @torch.no_grad()
    def compute_embeddings(image_batch, text_batch, attention_mask):
        image_batch = image_batch.to(device)
        text_batch = text_batch.to(device)
        attention_mask = attention_mask.to(device)

        return  model.get_image_features(image_batch).cpu(), model.get_text_features(text_batch, attention_mask).cpu()

    splits = datasets.get_dataset_split_names(args.dataset)
    dataset_splits = {}
    for split in splits:
        dataset_splits[split] = load_dataset(args.dataset, split=split).with_transform(image_preprocess)


    for key in dataset_splits:
        data_loader = torch.utils.data.DataLoader(dataset_splits[key], batch_size=512, shuffle=False, num_workers=8)
        image_embeddings = []
        text_embeddings = []
        for batch in tqdm(data_loader, desc = "Computing split: " + key):
            image_result, text_result = compute_embeddings(batch["pixel_values"],
                                                           batch["input_ids"],
                                                           batch["attention_mask"])
            image_embeddings.append(image_result)
            text_embeddings.append(text_result)

        embeddings_dict = {
            "images": torch.cat(image_embeddings),
            "texts": torch.cat(text_embeddings)
        }
        torch.save(embeddings_dict, f"{args.model.replace('/', '-')}__{args.dataset.replace('/', '-')}__{key}__embeddings.pt")


if __name__ == "__main__":
    main()
