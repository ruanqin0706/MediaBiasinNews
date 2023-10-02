import math
import pickle
from collections import defaultdict
from nltk.tokenize import word_tokenize


def calculate_label_counts(tuple_list):
    # Count label frequencies
    label_counts = defaultdict(int)
    for _, label in tuple_list:
        label_counts[label] += 1
    return label_counts


def calculate_word_scores(reviews, label_counts):
    # Tokenize the reviews into words
    words = [word_tokenize(review) for review, _ in reviews]

    # Flatten the word lists
    words = [word.lower() for sublist in words for word in sublist]

    # Count word frequencies
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1

    # Calculate co-occurrence frequencies
    cooccur_freq = defaultdict(int)
    pos_freq = defaultdict(int)
    neg_freq = defaultdict(int)
    for review, label in reviews:
        review_words = set(word_tokenize(review.lower()))
        for word in review_words:
            cooccur_freq[(word, label)] += 1
            if label == "True":
                pos_freq[word] += 1
            elif label == "False":
                neg_freq[word] += 1

    # Calculate PMI for words and store scores
    word_scores = {}
    total_words = len(words)
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


# Read data from a text file
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


if __name__ == '__main__':
    import time

    start_time = time.time()

    title_label_list = read_test_text(
        file_path="data/processed/hp_bypublisher_training_text.csv")

    word_scores = calculate_word_scores(title_label_list, calculate_label_counts(title_label_list))
    positive_words, negative_words = find_positive_negative_words(word_scores)
    # print(positive_words[:100])
    # print("*"*10)
    # print(negative_words[:100])

    # # Extract the first elements of each tuple
    # first_positive_elements = [tup[0] for tup in positive_words[:50]]
    # first_negative_elements = [tup[0] for tup in negative_words[:50]]
    # print(first_positive_elements)
    # print(first_negative_elements)
    with open("store/dict/pmi_biased_words.pkl", "wb") as f:
        pickle.dump(positive_words, f)

    with open("store/dict/pmi_neutral_words.pkl", "wb") as f:
        pickle.dump(negative_words, f)

    end_time = time.time()
    print(f"it costs: {(end_time - start_time) / 60} min.")
