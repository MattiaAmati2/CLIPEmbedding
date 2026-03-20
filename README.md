## CLIP Image and Text Embedding Extractor

To use this script, run python embedding_script.py --model [model] --dataset [dataset] --label_col [column_name]. <br>
The --model argument is required, and can be "openai/clip-vit-base-patch16" or "openai/clip-vit-base-patch32". <br>
The --dataset argument is also required, and it has to be a valid Hugging Face dataset ID. <br>
The --label-col argument is optional, and can be used to specify the name of the labels column

The output of this script is composed by one file for each split of the provided dataset, structured in a dictionary with the following columns:
- "class_names": a list of all the dataset's class names, sorted in ascending (alphabetical if these are strings) order
- "image_embeddings": the actual CLIP embeddings of the dataset's images
- "text_embeddings": the text embeddings of the class names described above
- "labels": the raw label corresponding to each image 

These files are saved in the same directory as the embedding_script.py file, as "[model]\__[dataset]\__[split]\__embeddings.pt"

## Implementation of basic classification techniques from the CLIP embeddings

Three basic classification methods are implemented, to find the actual accuracy of the CLIP model.
1. Zero-shot classification: python zero_shot_classification.py --filename [filename]
2. Nearest class mean few shot classification: python ncm_few_shot_classification.py --train_filename [filename] --test_filename [filename] --shot_number [shots]
3. A variation of the previous implementation, using Mahalanobis distance instead of cosine similiarity: python mahalanobis_distance_ncm.py --train_filename [filename] --test_filename [filename] --shot_number [shots]

All these commands require files produced using the previously described embedding script. 