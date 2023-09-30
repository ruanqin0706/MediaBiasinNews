import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re

FLAGS = re.MULTILINE | re.DOTALL


def re_sub(pattern, repl, text, flags=None):
    if flags is None:
        return re.sub(pattern, repl, text, flags=FLAGS)
    else:
        return re.sub(pattern, repl, text, flags=(FLAGS | flags))


def clean_txt(text):
    text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"&#160;", "", text)

    # Remove URL
    text = re_sub(r"(http)\S+", "", text)
    text = re_sub(r"(www)\S+", "", text)
    text = re_sub(r"(href)\S+", "", text)
    # Remove multiple spaces
    text = re_sub(r"[ \s\t\n]+", " ", text)

    # remove repetition
    text = re_sub(r"([!?.]){2,}", r"\1", text)
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", text)

    return text.strip()


def process():
    parser = argparse.ArgumentParser()

    parser.add_argument("--articles_file_path", type=str, default="../articles-training-bypublisher-20181122.xml")
    parser.add_argument("--labels_file_path", type=str, default="../ground-truth-training-bypublisher-20181122.xml")
    parser.add_argument("--write_path", type=str, default="../hp_bypublisher_training_text.csv")

    args = parser.parse_args()

    print(f"parse articles file: {args.articles_file_path}")
    articles_root = ET.parse(args.articles_file_path).getroot()
    print(f"parse labels file: {args.labels_file_path}")
    labels_root = ET.parse(args.labels_file_path).getroot()
    articles = articles_root.findall('article')
    labels = labels_root.findall('article')
    assert len(articles) == len(labels)

    with open(args.write_path, "wt") as f:
        for article, label in tqdm(zip(articles, labels), total=len(labels), desc="HP Detection xml2line"):
            article_id = article.get("id")  # article info
            label_id = label.get("id")  # label info
            assert article_id == label_id
            hyperpartisan = label.get("hyperpartisan")  # label info
            bias = label.get("bias")  # label info
            labeled_by = label.get("labeled-by")  # label info

            text = ET.tostring(article, method="text", encoding="utf-8").decode("utf-8")
            text = clean_txt(text)

            title = article.get("title")

            print(label_id, hyperpartisan, bias, labeled_by, text, title, sep="\t", file=f)


if __name__ == '__main__':
    process()
