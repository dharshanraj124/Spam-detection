import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import json
import os

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")[["v1","v2"]]
    df.columns = ["label","text"]
    df["text"] = df["text"].apply(clean)
    df["label"] = df["label"].map({"ham":0, "spam":1})
    return train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

def build_vocab(texts, max_words=5000):
    words = Counter()
    for t in texts:
        words.update(t.split())
    vocab = {w:i+1 for i,(w,_) in enumerate(words.most_common(max_words))}
    return vocab

def encode(text, vocab, max_len=150):
    seq = [vocab.get(w,0) for w in text.split()]
    seq = seq[:max_len] + [0]*(max_len-len(seq))
    return seq

def save_vocab(vocab, path="models/vocab.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)

def load_vocab(path="models/vocab.json"):
    with open(path, "r") as f:
        return json.load(f)
