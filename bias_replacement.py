import argparse
import os
import pickle
import time
import torch
from annoy import AnnoyIndex
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np

# Read data from a text file
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, DistilBertModel


def read_test_text(file_path):
    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            line_split = line.replace("\n", "").split("\t")

            news_title = line_split[-1]
            if 'true' == line_split[1]:
                news_label = "True"
            else:
                news_label = "False"
            data_list.append((news_title, news_label))
    return data_list


def read_test_text(file_path):
    with open(file_path, "rb") as f:
        data_list = pickle.load(f)
    return data_list


def cal_bert_emb(word):
    # Load the DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('new/local_distilbert/', local_files_only=True)
    model = DistilBertModel.from_pretrained('new/local_distilbert/', local_files_only=True)
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
    a = embeddings[:, 1:-1, :].numpy()
    b = np.mean(a, axis=1)
    c = b.reshape(b.shape[1])
    return c


def get_words(title_label_list,
              write_path,
              word2emb_path,
              required_pos=('ADJ', 'ADV', 'VERB', 'NOUN')):
    with open(word2emb_path, "rb") as f:
        ref_word2emb = pickle.load(f)

    word2emb = {}
    for title, label in title_label_list:
        sublist = word_tokenize(title.lower())
        for word, tag in pos_tag(sublist, tagset='universal'):
            if tag in required_pos:
                if word not in word2emb:
                    if word in ref_word2emb:
                        word2emb[word] = ref_word2emb[word]
                    else:
                        # calculate word embedding
                        embedding = cal_bert_emb(word)
                        embedding = embedding / np.linalg.norm(embedding)
                        word2emb[word] = embedding
    with open(write_path, "wb") as f:
        pickle.dump(word2emb, f)


def build_annoy_index(word2emb_path, save_path, k=10, n_trees=50):
    with open(word2emb_path, "rb") as f:
        word2emb = pickle.load(f)

    word_list = list(word2emb.keys())

    embedding_size = 768
    annoy_index = AnnoyIndex(embedding_size, metric='angular')
    for i, embedding in enumerate(word2emb.values()):
        annoy_index.add_item(i, embedding)
    annoy_index.build(n_trees=n_trees)
    annoy_index.save(save_path)


def get_annoy_index(embedding_size, annoy_path):
    annoy_index = AnnoyIndex(embedding_size, metric='angular')
    annoy_index.load(annoy_path)
    return annoy_index


def cal_sent_perplexity(sent_list, word_list, ):
    # Initialize the DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    text = "&&&&&DMTASK&&&&&".join(sent_list)

    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Calculate the perplexity for each token
    log_probs = []

    for i in range(1, len(input_ids)):
        input_id_tensor = torch.tensor(input_ids[:i]).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_id_tensor)
            logits = outputs.logits[0, -1]  # Logits for the last token in the sequence
            token_id = input_ids[i]
            log_prob = torch.nn.functional.log_softmax(logits, dim=0)[token_id]
            log_probs.append(log_prob.item())

    # Calculate PSN
    sentence_lengths = [len(sentence.split()) for sentence in text.split('&&&&&DMTASK&&&&&')]
    psn_scores = []
    for length, log_prob in zip(sentence_lengths, log_probs):
        psn_score = -log_prob / length
        psn_scores.append(psn_score)

    # Find the index of the maximum element
    max_index = psn_scores.index(max(psn_scores))
    return word_list[max_index]


def replace_words_in_text(title_list_path,
                          pmi_biased_words_path,
                          write_path,
                          word2emb_path,
                          annoy_index_path,
                          required_pos=('ADJ', 'ADV', 'VERB', 'NOUN'),
                          search_k=5):
    with open(title_list_path, "rb") as f:
        title_list = pickle.load(f)

    annoy_index = get_annoy_index(embedding_size=768,
                                  annoy_path=annoy_index_path)

    with open(word2emb_path, "rb") as f:
        word2emb = pickle.load(f)
    word_list = list(word2emb.keys())

    with open(pmi_biased_words_path, "rb") as f:
        pmi_biased_words_list = pickle.load(f)
    bias_word2pmi = {w: p for w, p, f in pmi_biased_words_list}

    replaced_text_list = []
    for title in title_list:
        title_new_list = []

        words = [word for word in word_tokenize(title.lower())]

        for idx, (word, tag) in enumerate(pos_tag(words, tagset='universal')):
            if word in word2emb and word in bias_word2pmi and tag in required_pos:
                similar_word_indices = annoy_index.get_nns_by_vector(word2emb[word], search_k + 1)
                similar_words = [word_list[idx] for idx in similar_word_indices[1:]]
                title_new_list.append(similar_words[0])
            else:
                title_new_list.append(word)

        title_new = " ".join(title_new_list)
        replaced_text_list.append(title_new)

    print(len(replaced_text_list))
    with open(write_path, "wb") as f:
        pickle.dump(replaced_text_list, f)


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", choices=['ALL', 'ADJ', 'ADV', 'VERB', 'NOUN'], default='VERB')
    parser.add_argument("--file_path", type=str, default='new/2017-07-032017-07-23.pkl')
    parser.add_argument("--write_path", type=str, default='new/emb/bert_emb')

    parser.add_argument("--pmi_bias_words_path", type=str, default='new/pmi/pmi_bias_words')
    parser.add_argument("--replaced_text_path", type=str, default='new/replace/replaced_text')
    parser.add_argument("--title_list_path", type=str, default="new/text/consistency.pkl")
    parser.add_argument("--word2emb_path", type=str, default="new/emb/bert_emb_ALL.pkl")
    parser.add_argument("--annoy_index_path", type=str, default="new/emb/word_annoy_t50_ALL.ann")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_params()
    # preparation work

    if args.pos == 'ALL':
        pos_set = ('ADJ', 'ADV', 'VERB', 'NOUN')
    elif args.pos == 'ADJ':
        pos_set = ('ADJ',)
    elif args.pos == 'ADV':
        pos_set = ('ADV',)
    elif args.pos == 'VERB':
        pos_set = ('VERB',)
    elif args.pos == 'NOUN':
        pos_set = ('NOUN',)
    else:
        1 / 0

    title_label_list = read_test_text(file_path=args.file_path)
    get_words(title_label_list=title_label_list,
              write_path=f"{args.write_path}_{args.pos}.pkl",
              word2emb_path="new/emb/bert_emb_ALL.pkl",
              required_pos=pos_set)

    build_annoy_index(word2emb_path=f"new/emb/bert_emb_{args.pos}.pkl",
                      save_path=f"new/emb/word_annoy_t50_{args.pos}.ann",
                      k=10,
                      n_trees=50)

    # replace work
    if args.pos == 'ALL':
        pos_set = ('ADJ', 'ADV', 'VERB', 'NOUN')
    elif args.pos == 'ADJ':
        pos_set = ('ADJ',)
    elif args.pos == 'ADV':
        pos_set = ('ADV',)
    elif args.pos == 'VERB':
        pos_set = ('VERB',)
    elif args.pos == 'NOUN':
        pos_set = ('NOUN',)
    else:
        1 / 0
    print(args.pos)

    replace_words_in_text(title_list_path=args.title_list_path,
                          pmi_biased_words_path=f"{args.pmi_bias_words_path}_{args.pos}.pkl",
                          write_path=f"{args.replaced_text_path}_{args.pos}.pkl",
                          word2emb_path=args.word2emb_path,
                          annoy_index_path=args.annoy_index_path,
                          required_pos=pos_set,
                          search_k=5)
