"""
NLP Course Project — Inference / Prediction
"""

import torch
import numpy as np
import re
import gensim.downloader as api
from model import SentimentClassifier, EMBEDDING_DIM

# ── Same constants as EMB.py ────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","love","loved",
    "best","fantastic","beautiful","superb","perfect","enjoy","enjoyed",
    "brilliant","awesome","favorite","fun","happy","nice",
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","worst","hate","hated","boring","poor",
    "waste","worse","horrible","disappointing","stupid","dull",
    "annoying","ridiculous","fail","failed","ugly","lame",
}
NEGATION_WORDS = {"not","no","never","nor","neither"}


def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def sentence_to_vector(tokens, model, dim=300):
    vectors, weights = [], []
    after_neg = False
    for token in tokens:
        if token in NEGATION_WORDS:
            after_neg = True
            continue
        if token in model:
            vectors.append(model[token])
            weights.append(0.25 if after_neg else 1.0)
        after_neg = False
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.average(np.asarray(vectors), axis=0,
                      weights=np.asarray(weights)).astype(np.float32)


def extra_features(tokens):
    n = max(len(tokens), 1)
    pos  = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg  = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    negs = sum(1 for t in tokens if t in NEGATION_WORDS)
    neg_pos = neg_neg = 0
    flipped = False
    for t in tokens:
        if t in NEGATION_WORDS:
            flipped = True; continue
        if flipped and t in POSITIVE_WORDS: neg_pos += 1
        if flipped and t in NEGATIVE_WORDS: neg_neg += 1
        flipped = False
    return np.array([
        np.log1p(n), pos/n, neg/n, negs/n,
        neg_pos/n, neg_neg/n,
        (pos - neg + neg_neg - neg_pos)/n,
        (pos + neg)/n,
    ], dtype=np.float32)


def get_vector(text, glove):
    tokens = preprocess(text)
    return np.concatenate([
        sentence_to_vector(tokens, glove),
        extra_features(tokens),
    ])


def predict(text, model, glove, device):
    vec = get_vector(text, glove)
    x   = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prob = model(x).item()
    label = "Positive" if prob >= 0.5 else "Negative"
    bar   = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
    print(f"\n  '{text}'")
    print(f"  [{bar}] {prob*100:.1f}%")
    print(f"  -> {label}")


# ══════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print(" Loading GloVe...")
    glove = api.load("glove-wiki-gigaword-300")

    print(" Loading trained model...")
    model = SentimentClassifier().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print(" Model loaded!\n")

    # ── Test sentences ──────────────────────────────────
    test_sentences = [
        "this movie is amazing",
        "this movie is terrible",
        "this movie is not bad",      # <- Negation
        "not good at all",
        "I love this film",
        "I hate this film",
        "the acting was brilliant but the story was boring",
    ]

    print("=" * 50)
    print("        Trained model results")
    print("=" * 50)
    for s in test_sentences:
        predict(s, model, glove, device)

    # ── Interactive mode ──────────────────────────────────
    print("\n" + "=" * 50)
    print("Type a sentence and see the result (type 'exit' to quit)")
    print("=" * 50)
    while True:
        text = input("\n Type a sentence: ").strip()
        if text.lower() == "exit":
            break
        if text:
            predict(text, model, glove, device)
