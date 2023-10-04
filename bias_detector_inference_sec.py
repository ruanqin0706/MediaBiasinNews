import argparse
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np


# original text, processed_text [a list of text]
def read_text(text_file_path):
    with open(text_file_path, "rb") as f:
        text_list = pickle.load(f)

    return text_list


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

    parser.add_argument("--mode", choices=["processed", "originals"], default="processed")
    parser.add_argument("--pos", choices=['ALL', 'ADJ', 'ADV', 'VERB', 'NOUN'], default='NOUN')

    parser.add_argument("--text_file_path", type=str,
                        default="new/replace/replaced_text")
    # default="new/text/consistency.pkl")

    # sent emb saved path
    parser.add_argument("--save_path", type=str,
                        default="new/sent_comparison/sent")

    return parser.parse_args()


args = parse_params()
tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
batch_size = args.bs
max_length = args.max_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.mode == "originals":
    test_texts = read_text(text_file_path=f"{args.text_file_path}")  # original
elif args.mode == "processed":
    test_texts = read_text(text_file_path=f"{args.text_file_path}_{args.pos}.pkl")  # processed text
else:
    1 / 0
print(f"current mode is: {args.mode}")

test_dataset = TextClassificationDataset(test_texts, [0] * len(test_texts), tokenizer,
                                         max_length)  # Labels don't matter for inference
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
# best_model.load_state_dict(torch.load(args.checkpoint_path))
best_model.to(device)

best_model.eval()

# Initialize a list to store sentence emb
sentence_embedding_list = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = best_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        # Extract the embeddings for the [CLS] token (first token) of each sentence
        sentence_embedding = last_hidden_states[:, 0, :].cpu().numpy()  # (bs, 768)
        sentence_embedding_list.append(sentence_embedding)

sentence_emb_arr = np.concatenate(sentence_embedding_list, axis=0)
print(sentence_emb_arr.shape)
if args.mode == "originals":
    real_path = f"{args.save_path}.npy"
elif args.mode == "processed":
    real_path = f"{args.save_path}_{args.pos}.npy"
else:
    1 / 0
with open(real_path, "wb") as f:
    np.save(f, sentence_emb_arr)
