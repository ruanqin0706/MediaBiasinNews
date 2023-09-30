import argparse
import glob
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np


# Read data from a text file
def read_test_text(file_path):
    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            line_split = line.replace("\n", "").split("\t")

            news_title = line_split[-1]
            if 'true' == line_split[1]:
                data_list.append(news_title)
    print(len(data_list))
    return data_list


# Define a custom dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=50)

    parser.add_argument("--data_file_path", type=str,
                        default="/Users/qin/phd_source/MediaBiasinNews/data/processed/hp_bypublisher_training_text.csv")

    # For storing
    parser.add_argument("--save_dir", type=str,
                        default="/Users/qin/phd_source/MediaBiasinNews/store/checkpoint")
    return parser.parse_args()


args = parse_params()
tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
batch_size = args.bs
max_length = args.max_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference code
# test_texts = ["This is a news title to classify.", "Another news title to classify.",
#               ]  # Pseudo code for testing

with open("/Users/qin/phd_source/MediaBiasinNews/store/replacement/replaced_text.pkl", "rb") as f:
    test_texts = pickle.load(f)

# test_texts = read_test_text(file_path=args.data_file_path)
test_dataset = TextClassificationDataset(test_texts, [0] * len(test_texts), tokenizer,
                                         max_length)  # Labels don't matter for inference
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_checkpoints = []
for checkpoint_path in sorted(glob.glob(os.path.join(args.save_dir, "*.pth"))):
    print("Load ckpt: ", checkpoint_path)
    # Load the best checkpoint for inference
    best_checkpoint = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    best_checkpoint.load_state_dict(torch.load(checkpoint_path))
    best_checkpoint.to(device)
    best_checkpoints.append(best_checkpoint)

print(len(best_checkpoints))
for fold, best_model in enumerate(best_checkpoints):
    # Initialize a list to store predicted probabilities
    all_probabilities = []

    best_model.eval()
    fold_probabilities = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = best_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = nn.functional.softmax(logits, dim=1)
            fold_probabilities.extend(probabilities[:, 1].cpu().tolist())

    all_probabilities.append(fold_probabilities)

    # Calculate the average probabilities from all folds
    average_probabilities = np.mean(all_probabilities, axis=0)

    # print("Predicted Probabilities:")
    # print(average_probabilities)

    count = np.count_nonzero(average_probabilities <= 0.5)

    print("Number of elements less than or equal to the threshold:", count)

    print("*"*10)