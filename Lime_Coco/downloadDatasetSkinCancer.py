from datasets import load_dataset
import pandas as pd
ds = load_dataset("marmal88/skin_cancer")

train_dataset = load_dataset("marmal88/skin_cancer", split="train")
valid_dataset = load_dataset("marmal88/skin_cancer", split="validation")
test_dataset  = load_dataset("marmal88/skin_cancer", split="test")


# print(len(train_dataset))
# print(train_dataset[0])
# print(len(valid_dataset))
# print(valid_dataset[0])
# print(len(test_dataset))
# print(test_dataset[0])