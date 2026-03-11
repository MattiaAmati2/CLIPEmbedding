## CLIP Image and Text Embedding Extractor

To use this script, run python embedding_script.py --model [model] --dataset [dataset]. <br>
The --model argument is required, and can be "openai/clip-vit-base-patch16" or "openai/clip-vit-base-patch32". <br>
The --dataset argument is also required, and it has to be a valid Hugging Face dataset ID. <br>
The resulting embeddings are saved as dictionaries in a "model__dataset__split__embeddings.pt" file in the same directory as the embedding script, and the auxiliary display_shape.py script can print their shape, with python display_shape.py --filename [filename].