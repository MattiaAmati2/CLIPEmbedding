import argparse

import datasets
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices = ["openai/clip-vit-base-patch16",
                                                                                   "openai/clip-vit-base-patch32"])
    parser.add_argument("--dataset", type=str, required=True)

    args : argparse.Namespace = parser.parse_args()

    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)

    def image_preprocess(batch):
        pass

    splits = datasets.get_dataset_split_names(args.dataset)
    dataset_splits = {}
    for split in splits:
        dataset_splits[split] = load_dataset(args.dataset, split=split).with_transform(image_preprocess)



if __name__ == "__main__":
    main()
