import argparse
import os

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import random
import numpy as np
import torch

seed = 2023

# Set seed for Python's random module
random.seed(seed)

# Set seed for NumPy
np.random.seed(seed)

# Set seed for PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Read data from a text file
def read_text(file_path):
    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            line_split = line.replace("\n", "").split("\t")

            news_title = line_split[-1]
            if 'true' == line_split[1]:
                news_label = 0
            else:
                news_label = 1

            data_list.append((news_title, news_label))
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


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--k_fold", type=int, default=5)

    # for-loop
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=32)

    # for data
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--data_file_path", type=str,
                        default="/Users/qin/phd_source/MediaBiasinNews/data/processed/hp_bypublisher_training_text.csv")

    # For storing
    parser.add_argument("--save_dir", type=str,
                        default="/Users/qin/phd_source/MediaBiasinNews/store/checkpoint")
    return parser.parse_args()


# Define the model and optimizer
args = parse_params()
# # Pseudo code for running test
# dataset = [("Politician Accused of Corruption in Recent Scandal", 1),
#            ("Partisan Clash Erupts in Heated Debate", 1),
#            ("Sensational Claims Shake Up Local Election", 1),
#            ("Controversial Policy Sparks Outrage Amongst Critics", 1),
#            ("Media Outlet Faces Backlash for Biased Reporting", 1),
#            ("Economic Report Indicates Stable Growth", 0),
#            ("Scientists Make Breakthrough in Medical Research", 0),
#            ("Community Volunteers Clean Up Local Park", 0),
#            ("International Diplomats Work Towards Peaceful Resolution", 0),
#            ("Weather Forecast Predicts Sunny Days Ahead", 0)]
dataset = read_text(file_path=args.data_file_path)

model_name = args.model_name
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 for binary classification

# Define k-fold cross-validation
k = args.k_fold  # Number of folds
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

# Define the params during the for-loop and others
num_epochs = args.num_epochs
batch_size = args.bs
max_length = args.max_len

# Perform k-fold cross-validation
fold_accuracies = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a list to store the best model checkpoints
os.makedirs(args.save_dir, exist_ok=True)
best_checkpoints = []
avg_best_accuracy = 0
for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(dataset)), [label for _, label in dataset])):
    print(f"Fold {fold + 1}/{k}")

    train_texts = [dataset[i][0] for i in train_idx]
    train_labels = [dataset[i][1] for i in train_idx]
    val_texts = [dataset[i][0] for i in val_idx]
    val_labels = [dataset[i][1] for i in val_idx]

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_accuracy = 0

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

        # Evaluate on the validation set
        accuracy = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Check if the current model is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save the current model checkpoint
            checkpoint_path = os.path.join(args.save_dir, f"best_model_checkpoint_fold_{fold}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    avg_best_accuracy += best_accuracy
    # Load the best checkpoint for inference
    best_checkpoint = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    best_checkpoint.load_state_dict(torch.load(checkpoint_path))
    best_checkpoint.to(device)
    best_checkpoints.append(best_checkpoint)

print("average best accuracy:", avg_best_accuracy / k)
