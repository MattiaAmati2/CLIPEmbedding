import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type = str)
args = parser.parse_args()

data = torch.load(args.filename)
for key in data.keys():
    print(key, data[key].shape)
