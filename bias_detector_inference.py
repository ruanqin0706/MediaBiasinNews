import argparse
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

    # For storing
    parser.add_argument("--checkpoint_path", type=str,
                        default="store/checkpoint_1/best_model_checkpoint_fold_2.pth")

    parser.add_argument("--action_consistency_check", action="store_true", default=True)
    parser.add_argument("--consistency_path", type=str,
                        default="data/processed/consistency.npy")

    parser.add_argument("--mode", choices=["processed", "originals"], default="processed")

    parser.add_argument("--processed_file_path", type=str,
                        default="store/replacement/replaced_text.pkl")
    parser.add_argument("--data_file_path", type=str,
                        default="data/processed/hp_bypublisher_training_text.csv")

    return parser.parse_args()


args = parse_params()
tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
batch_size = args.bs
max_length = args.max_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.mode == "processed":
    print("processed")
    # processed text
    with open("/Users/qin/phd_source/MediaBiasinNews/store/replacement/replaced_text.pkl", "rb") as f:
        test_texts = pickle.load(f)
else:
    print("originals")
    # original text
    test_texts = read_test_text(file_path=args.data_file_path)

if args.action_consistency_check:
    with open(args.consistency_path, "rb") as f:
        consistency_idx_arr = np.load(f)
else:
    consistency_idx_arr = None

test_dataset = TextClassificationDataset(test_texts, [0] * len(test_texts), tokenizer,
                                         max_length)  # Labels don't matter for inference
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
best_model.load_state_dict(torch.load(args.checkpoint_path))
best_model.to(device)

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
        fold_probabilities.extend(probabilities[:, 0].cpu().tolist())

all_probabilities.append(fold_probabilities)

# Calculate the average probabilities from all folds
average_probabilities = np.mean(all_probabilities, axis=0)

if isinstance(consistency_idx_arr, np.ndarray):
    consistency_idx_arr = consistency_idx_arr.astype(np.int32)
    average_probabilities = average_probabilities[consistency_idx_arr]

print(f"the average probability under {args.mode} mode is: ", np.average(average_probabilities))
