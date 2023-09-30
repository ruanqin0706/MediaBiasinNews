import os
import pickle
import time

from annoy import AnnoyIndex
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from gensim.models import KeyedVectors
import numpy as np
# nltk.download('universal_tagset')
import pandas as pd


# data
# news.tsv
# training data (behavior files)
# observation data

# training algorithms
#

# 训练词典，识别bias polarity of words
#


# def find_similarity_words(word):
#     pass
#     # return word_list
#
# def check_polarity(word):
#     return True
#
# def process_text(line):
#     pass
#     # return [(word, pos)]
#
# def replace_words():
#     pass
#
#     pos_exempt_set = set()
#
#     # {nid: title}
#     nid2title = dict()
#     for nid, line in nid2title.items():
#         tuple_list = process_text(line)
#         for word, pos in tuple_list:
#             if pos not in pos_exempt_set and check_polarity(word):


# 如何替换词。how to replace words?
# 1.对词语遍历， 如果词语是biased，就去找和它相似的词语（用similarity embedding）,
# 再做过滤，需要相同的词性，然后区分biased还是neutral，如果存在neutral的，按照分数选择一个，如过都是biased，按照分数选择比当前词低的
# # 涉及一个问题，要不要替换所有的title？

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


def get_words(title_label_list_path, emb_path, word_path, pre_train_dict_path, write_path):
    title_label_list = read_test_text(title_label_list_path)

    start_time = time.time()
    emb_arr = np.load(emb_path)
    with open(word_path, "rb") as f:
        word_dict = pickle.load(f)
    train_dict = dict()
    for word, idx in word_dict.items():
        train_dict[word] = emb_arr[idx]
    end_time = time.time()
    print(f"it costs: {(end_time - start_time) / 60} min")

    start_time = time.time()
    pre_train_model = KeyedVectors. \
        load_word2vec_format(pre_train_dict_path, binary=True)
    end_time = time.time()
    print(f"it costs: {(end_time - start_time) / 60} min")

    num_hits_train = 0
    num_hits_pre = 0
    num_miss = 0
    word2emb = {}
    for title, label in title_label_list:
        words = [word.lower() for word in word_tokenize(title)]
        for word, tag in pos_tag(words, tagset='universal'):
            if word not in word2emb:
                if word in train_dict:
                    num_hits_train += 1
                    embedding = train_dict[word]
                    embedding = embedding / np.linalg.norm(embedding)
                    word2emb[word] = embedding
                elif word in pre_train_model.wv:
                    num_hits_pre += 1
                    embedding = pre_train_model.wv[word]
                    embedding = embedding / np.linalg.norm(embedding)
                    word2emb[word] = embedding
                else:
                    num_miss += 1

    print(num_hits_train, num_hits_pre, num_miss, (num_hits_train + num_hits_pre + num_miss))
    with open(write_path, "wb") as f:
        pickle.dump(word2emb, f)
    print(f"it contains: {len(word2emb)}")


def build_annoy_index(word2emb_path, save_dir, k=10, n_trees=50):
    with open(word2emb_path, "rb") as f:
        word2emb = pickle.load(f)

    word_list = list(word2emb.keys())

    embedding_size = 300
    annoy_index = AnnoyIndex(embedding_size, metric='angular')
    for i, embedding in enumerate(word2emb.values()):
        annoy_index.add_item(i, embedding)
    annoy_index.build(n_trees=n_trees)
    annoy_index.save(os.path.join(save_dir, f"word_annoy_t{n_trees}.ann"))

    # similar_word_indices = annoy_index.get_nns_by_vector(word2emb['ugly'], k+1)
    #
    # similar_words = [word_list[idx] for idx in similar_word_indices[1:]]
    # print(f"Top-{k} similar words to 'like': {similar_words}")


def get_annoy_index(embedding_size, annoy_path):
    annoy_index = AnnoyIndex(embedding_size, metric='angular')
    annoy_index.load(annoy_path)
    return annoy_index


def replace_words_in_text(word2emb_path, annoy_index_path, title_label_list_path,
                          pmi_biased_words_path,
                          pmi_neutral_words_path,
                          write_path):
    with open(pmi_biased_words_path, "rb") as f:
        pmi_biased_words_list = pickle.load(f)
    with open(pmi_neutral_words_path, "rb") as f:
        pmi_neutral_words_list = pickle.load(f)

    biased_word2pmi = {w: p for w, p, f in pmi_biased_words_list}
    neutral_word2pmi = {w: p for w, p, f in pmi_neutral_words_list}

    # news_path,
    # news_df = pd.read_csv(news_path, header=None, sep="\t")
    with open(word2emb_path, "rb") as f:
        word2emb = pickle.load(f)
    word_list = list(word2emb.keys())

    annoy_index = get_annoy_index(embedding_size=300,
                                  annoy_path=annoy_index_path)

    title_label_list = read_test_text(file_path=title_label_list_path)

    replaced_text_list = []
    for title, label in title_label_list:
        if label == "True":
            title_new_list = []

            words = [word.lower() for word in word_tokenize(title)]

            for word, tag in pos_tag(words, tagset='universal'):
                if word in word2emb and word in biased_word2pmi:
                    similar_word_indices = annoy_index.get_nns_by_vector(word2emb[word], 10 + 1)
                    similar_words = [word_list[idx] for idx in similar_word_indices[1:]]
                    title_new_list.append(similar_words[0])
                else:
                    title_new_list.append(word)

            title_new = " ".join(title_new_list)
            replaced_text_list.append(title_new)
            # print(title)
            # print(title_new)
            # print("*"*10)

    print(len(replaced_text_list))
    with open(write_path, "wb") as f:
        pickle.dump(replaced_text_list, f)


if __name__ == '__main__':
    # get_words(title_label_list_path="/Users/qin/phd_source/MediaBiasinNews/data/processed/hp_bypublisher_training_text.csv",
    #           emb_path="/Users/qin/Downloads/data0404/MINDlarge_utils/embedding.npy",
    #           word_path="/Users/qin/Downloads/data0404/MINDlarge_utils/word_dict.pkl",
    #           pre_train_dict_path="/Users/qin/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz",
    #           write_path="/Users/qin/phd_source/MediaBiasinNews/store/replacement/word2emb.pkl")
    #
    # build_annoy_index(word2emb_path="/Users/qin/phd_source/MediaBiasinNews/store/replacement/word2emb.pkl",
    #                   save_dir="/Users/qin/phd_source/MediaBiasinNews/store/replacement",
    #                   k=10,
    #                   n_trees=50)

    replace_words_in_text(word2emb_path="/Users/qin/phd_source/MediaBiasinNews/store/replacement/word2emb.pkl",
                          annoy_index_path="/Users/qin/phd_source/MediaBiasinNews/store/replacement/word_annoy_t50.ann",
                          title_label_list_path="/Users/qin/phd_source/MediaBiasinNews/data/processed/hp_bypublisher_training_text.csv",
                          pmi_biased_words_path="/Users/qin/phd_source/MediaBiasinNews/store/dict/pmi_biased_words.pkl",
                          pmi_neutral_words_path="/Users/qin/phd_source/MediaBiasinNews/store/dict/pmi_neutral_words.pkl",
                          write_path="/Users/qin/phd_source/MediaBiasinNews/store/replacement/replaced_text.pkl")
