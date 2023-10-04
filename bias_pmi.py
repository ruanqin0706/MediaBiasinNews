import argparse
import math
import pickle
from collections import defaultdict

from nltk import pos_tag
from nltk.tokenize import word_tokenize


def calculate_label_counts(tuple_list):
    # Count label frequencies
    label_counts = defaultdict(int)
    for _, label in tuple_list:
        label_counts[label] += 1
    return label_counts


def calculate_word_scores(reviews, label_counts, required_pos=('ADJ', 'ADV', 'VERB', 'NOUN')):
    cooccur_freq = defaultdict(int)
    pos_freq = defaultdict(int)
    neg_freq = defaultdict(int)
    total_word_list = []
    for review, label in reviews:
        sublist = word_tokenize(review.lower())
        word_list = []
        for word, tag in pos_tag(sublist, tagset='universal'):
            if tag in required_pos:
                word_list.append(word)

        review_words = set(word_list)
        for word in review_words:
            cooccur_freq[(word, label)] += 1
            if label == "True":
                pos_freq[word] += 1
            elif label == "False":
                neg_freq[word] += 1

        total_word_list.extend(word_list)

    # Count word frequencies
    word_freq = defaultdict(int)
    for word in total_word_list:
        word_freq[word] += 1

    # Calculate PMI for words and store scores
    word_scores = {}
    total_words = len(total_word_list)
    for word, freq in word_freq.items():
        if len(word) > 2:
            p_word = freq / total_words
            pmi_pos = 0.0
            pmi_neg = 0.0
            if pos_freq[word] > 0:
                p_word_pos = pos_freq[word] / total_words
                pmi_pos = math.log(p_word_pos / (p_word * (label_counts["True"] / total_words)))
            if neg_freq[word] > 0:
                p_word_neg = neg_freq[word] / total_words
                pmi_neg = math.log(p_word_neg / (p_word * (label_counts["False"] / total_words)))
            word_scores[word] = (freq, pmi_pos, pmi_neg)

    return word_scores


def find_positive_negative_words(word_scores):
    positive_words = []
    negative_words = []

    for word, (freq, pmi_pos, pmi_neg) in word_scores.items():
        if freq > 0:
            if pmi_pos > pmi_neg:
                positive_words.append((word, pmi_pos, freq))
            elif pmi_neg > pmi_pos:
                negative_words.append((word, pmi_neg, freq))

    positive_words.sort(key=lambda x: x[1], reverse=True)
    negative_words.sort(key=lambda x: x[1], reverse=True)

    return positive_words, negative_words


def read_test_text(file_path):
    with open(file_path, "rb") as f:
        data_list = pickle.load(f)
    return data_list


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='new/2017-07-032017-07-23.pkl')
    parser.add_argument("--pos", choices=['ALL', 'ADJ', 'ADV', 'VERB', 'NOUN'], default='ADV')
    parser.add_argument("--pmi_bias_words_path", type=str, default='new/pmi/pmi_bias_words')
    parser.add_argument("--pmi_neutral_words_path", type=str, default='new/pmi/pmi_neutral_words')
    return parser.parse_args()


if __name__ == '__main__':
    import time

    start_time = time.time()

    args = parse_params()

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

    title_label_list = read_test_text(
        file_path=args.file_path)

    word_scores = calculate_word_scores(title_label_list,
                                        calculate_label_counts(title_label_list),
                                        required_pos=pos_set)
    positive_words, negative_words = find_positive_negative_words(word_scores)
    # print(positive_words[:100])
    # print("*"*10)
    # print(negative_words[:100])

    # # Extract the first elements of each tuple
    # first_positive_elements = [tup[0] for tup in positive_words[:50]]
    # first_negative_elements = [tup[0] for tup in negative_words[:50]]
    # print(first_positive_elements)
    # print(first_negative_elements)
    with open(f"{args.pmi_bias_words_path}_{args.pos}.pkl", "wb") as f:
        pickle.dump(positive_words, f)

    with open(f"{args.pmi_neutral_words_path}_{args.pos}.pkl", "wb") as f:
        pickle.dump(negative_words, f)

    end_time = time.time()
    print(f"it costs: {(end_time - start_time) / 60} min.")
