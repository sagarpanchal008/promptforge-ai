"""
==============================================================================
FILE: utils/preprocessing.py
PURPOSE:
    Production NLP preprocessing pipeline for the Intelligent Prompt
    Optimization Assistant.
    Implements text cleaning, contraction expansion, vocabulary tokenization,
    sequence padding, label encoding, and stratified train/val/test splitting.

HOW IT CONNECTS TO THE PROJECT:
    - Ingests 'dataset/prompts.csv' (from Phase 1).
    - Outputs 'dataset/processed_dataset.csv' and 'dataset/preprocessed_splits.json'.
    - Serializes 'models/tokenizer.pkl' and 'models/label_encoder.pkl'.
    - Supplies preprocessed sequence matrices for Phase 3 (1D-CNN) and
      Phase 4 (Bi-LSTM) training.
    - Used at inference time by Agent 2 (DL Evaluator) in Phase 6 and
      the Streamlit frontend in Phase 8.

KEY ACADEMIC PRINCIPLE:
    Zero Data Leakage: The tokenizer and vocabulary are fitted EXCLUSIVELY
    on the training set (X_train), and then applied to transform validation
    and test sets (X_val, X_test).

EXECUTION:
    python utils/preprocessing.py
==============================================================================
"""

import csv
import json
import os
import pickle
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_CSV = os.path.join(PROJECT_ROOT, "dataset", "prompts.csv")
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed_dataset.csv")
SPLITS_JSON = os.path.join(PROJECT_ROOT, "dataset", "preprocessed_splits.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TOKENIZER_PKL = os.path.join(MODELS_DIR, "tokenizer.pkl")
LABEL_ENCODER_PKL = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Configuration Constants
DEFAULT_MAX_WORDS = 5000       # Maximum vocabulary size
DEFAULT_MAX_LEN = 120          # Maximum token sequence length for CNN & LSTM padding
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# 1. CONTRACTION EXPANSION MAP
# -----------------------------------------------------------------------------
# Standardizing contractions prevents duplicate vocabulary tokens
# (e.g., "don't" vs "do not").
CONTRACTION_MAP = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "might've": "might have", "must've": "must have",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}


# -----------------------------------------------------------------------------
# 2. TEXT CLEANING FUNCTIONS
# -----------------------------------------------------------------------------

def expand_contractions(text: str) -> str:
    """Replaces informal English contractions with their expanded forms."""
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
        flags=re.IGNORECASE,
    )

    def replace_match(match):
        matched_text = match.group(0).lower()
        return CONTRACTION_MAP.get(matched_text, matched_text)

    return pattern.sub(replace_match, text)


def clean_text(text: str) -> str:
    """
    Performs comprehensive NLP text normalization:
    1. Expands contractions ("can't" -> "cannot")
    2. Converts to lowercase
    3. Normalizes whitespace and removes unwanted non-ASCII characters
    4. Retains meaningful alphanumeric characters and standard punctuation
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Contraction expansion
    text = expand_contractions(text)

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Remove URLs if any
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Step 4: Remove Markdown code block backticks and formatting markers
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"[`*_{}\[\]()#+\-!]", " ", text)

    # Step 5: Replace multiple spaces and newlines with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------------------------------------------------
# 3. TOKENIZER & SEQUENCE PADDING
# -----------------------------------------------------------------------------

class TextTokenizer:
    """
    Custom NLP Tokenizer compatible with Keras Tokenizer architecture.
    Maps words to integer indices, handles <PAD> (index 0) and <UNK> (index 1),
    and pads sequences to a fixed sequence length for CNN & LSTM input.
    """

    def __init__(self, max_words: int = 5000, oov_token: str = "<UNK>"):
        self.max_words = max_words
        self.oov_token = oov_token
        self.word_index: Dict[str, int] = {"<PAD>": 0, self.oov_token: 1}
        self.index_word: Dict[int, str] = {0: "<PAD>", 1: self.oov_token}
        self.word_counts: Counter = Counter()
        self.vocab_size: int = 2

    def fit_on_texts(self, texts: List[str]):
        """
        Builds the vocabulary from a list of clean text documents.
        Sorted by frequency, capped at max_words.
        """
        for text in texts:
            words = text.split()
            self.word_counts.update(words)

        # Most common words up to max_words - 2 (leaving room for <PAD> and <UNK>)
        most_common = self.word_counts.most_common(self.max_words - 2)
        idx = 2
        for word, _ in most_common:
            self.word_index[word] = idx
            self.index_word[idx] = word
            idx += 1

        self.vocab_size = len(self.word_index)

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Converts raw cleaned text into lists of integer token IDs."""
        sequences = []
        unk_idx = self.word_index.get(self.oov_token, 1)
        for text in texts:
            words = text.split()
            seq = [self.word_index.get(w, unk_idx) for w in words]
            sequences.append(seq)
        return sequences

    def pad_sequences(
        self,
        sequences: List[List[int]],
        maxlen: int = DEFAULT_MAX_LEN,
        padding: str = "post",
        truncating: str = "post",
    ) -> List[List[int]]:
        """
        Pads or truncates integer sequences to a uniform length (maxlen).
        'post' padding appends zeros at the end of the sequence.
        """
        padded = []
        for seq in sequences:
            if len(seq) > maxlen:
                if truncating == "post":
                    seq = seq[:maxlen]
                else:
                    seq = seq[-maxlen:]
            else:
                pad_len = maxlen - len(seq)
                if padding == "post":
                    seq = seq + [0] * pad_len
                else:
                    seq = [0] * pad_len + seq
            padded.append(seq)
        return padded


# -----------------------------------------------------------------------------
# 4. LABEL ENCODER
# -----------------------------------------------------------------------------

class QualityLabelEncoder:
    """
    Encodes categorical prompt quality labels ('Poor', 'Average', 'Good')
    into integers (0, 1, 2) and provides one-hot encoding for Softmax loss.
    """

    def __init__(self):
        self.classes_: List[str] = ["Poor", "Average", "Good"]
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_class: Dict[int, str] = {i: c for i, c in enumerate(self.classes_)}

    def transform(self, labels: List[str]) -> List[int]:
        """Maps strings to integers [0, 1, 2]."""
        return [self.class_to_idx[lbl] for lbl in labels]

    def inverse_transform(self, indices: List[int]) -> List[str]:
        """Maps integers back to original class strings."""
        return [self.idx_to_class[idx] for idx in indices]

    def to_one_hot(self, indices: List[int]) -> List[List[float]]:
        """Converts integer class indices into 3-dimensional one-hot vectors."""
        num_classes = len(self.classes_)
        one_hot = []
        for idx in indices:
            vec = [0.0] * num_classes
            vec[idx] = 1.0
            one_hot.append(vec)
        return one_hot


# -----------------------------------------------------------------------------
# 5. STRATIFIED TRAIN / VAL / TEST SPLITTER
# -----------------------------------------------------------------------------

def stratified_split(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Performs a deterministic stratified split based on 'quality_label'
    to ensure identical class distributions across Train, Validation, and Test.
    """
    import random
    rng = random.Random(seed)

    # Group records by class
    by_class: Dict[str, List[Dict]] = {}
    for r in records:
        lbl = r["quality_label"]
        by_class.setdefault(lbl, []).append(r)

    train_set, val_set, test_set = [], [], []

    for lbl, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set.extend(group[:n_train])
        val_set.extend(group[n_train : n_train + n_val])
        test_set.extend(group[n_train + n_val :])

    # Shuffle combined sets
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    return train_set, val_set, test_set


# -----------------------------------------------------------------------------
# 6. PIPELINE ORCHESTRATION & RUNNER
# -----------------------------------------------------------------------------

def run_preprocessing_pipeline():
    """
    Executes the end-to-end preprocessing workflow:
    1. Loads dataset/prompts.csv
    2. Applies text cleaning
    3. Fits Tokenizer on Train set ONLY (Zero Data Leakage)
    4. Pads sequences to max_len = 120
    5. Encodes quality labels
    6. Saves processed_dataset.csv, tokenizer.pkl, label_encoder.pkl, and preprocessed_splits.json
    """
    print("=" * 75)
    print("PHASE 2: NLP PREPROCESSING & TOKENIZATION PIPELINE")
    print("=" * 75)

    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] Raw dataset not found at {DATASET_CSV}. Run Phase 1 first.")
        sys.exit(1)

    # Step 1: Load raw dataset
    raw_records = []
    with open(DATASET_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_records.append(row)

    print(f"[*] Step 1: Ingested {len(raw_records)} raw records from dataset/prompts.csv")

    # Step 2: Clean prompt texts
    for row in raw_records:
        row["clean_prompt"] = clean_text(row["prompt"])

    # Step 3: Stratified Train / Validation / Test Splitting (70% / 15% / 15%)
    train_records, val_records, test_records = stratified_split(
        raw_records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED
    )

    print(f"[*] Step 2: Stratified Split Complete:")
    print(f"    - Training Set   : {len(train_records):>4} samples (70.0%)")
    print(f"    - Validation Set : {len(val_records):>4} samples (15.0%)")
    print(f"    - Test Set       : {len(test_records):>4} samples (15.0%)")

    # Step 4: Tokenization (Zero Data Leakage: fit on training data only)
    train_texts = [r["clean_prompt"] for r in train_records]
    tokenizer = TextTokenizer(max_words=DEFAULT_MAX_WORDS)
    tokenizer.fit_on_texts(train_texts)

    print(f"[*] Step 3: Fitted Tokenizer Vocabulary (Training Set Only):")
    print(f"    - Vocabulary Size : {tokenizer.vocab_size} unique tokens (including <PAD> & <UNK>)")
    print(f"    - Sequence Padding: Uniform {DEFAULT_MAX_LEN} tokens (post-padded)")

    # Step 5: Convert to sequences & pad
    X_train_seq = tokenizer.pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=DEFAULT_MAX_LEN)
    X_val_seq = tokenizer.pad_sequences(tokenizer.texts_to_sequences([r["clean_prompt"] for r in val_records]), maxlen=DEFAULT_MAX_LEN)
    X_test_seq = tokenizer.pad_sequences(tokenizer.texts_to_sequences([r["clean_prompt"] for r in test_records]), maxlen=DEFAULT_MAX_LEN)

    # Step 6: Label Encoding
    label_encoder = QualityLabelEncoder()
    y_train = label_encoder.transform([r["quality_label"] for r in train_records])
    y_val = label_encoder.transform([r["quality_label"] for r in val_records])
    y_test = label_encoder.transform([r["quality_label"] for r in test_records])

    # Convert to one-hot vectors for Softmax output in CNN & LSTM
    y_train_one_hot = label_encoder.to_one_hot(y_train)
    y_val_one_hot = label_encoder.to_one_hot(y_val)
    y_test_one_hot = label_encoder.to_one_hot(y_test)

    # Step 7: Serialize Tokenizer and Label Encoder to models/
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(TOKENIZER_PKL, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(LABEL_ENCODER_PKL, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"[+] Step 4: Serialized Preprocessing Artifacts:")
    print(f"    - {TOKENIZER_PKL}")
    print(f"    - {LABEL_ENCODER_PKL}")

    # Step 8: Save preprocessed dataset CSV
    fieldnames = list(raw_records[0].keys())
    with open(PROCESSED_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_records)

    print(f"[+] Step 5: Saved Cleaned Dataset to: {PROCESSED_CSV}")

    # Step 9: Save preprocessed numerical splits for Phase 3 (CNN) & Phase 4 (LSTM)
    splits_payload = {
        "metadata": {
            "vocab_size": tokenizer.vocab_size,
            "max_len": DEFAULT_MAX_LEN,
            "classes": label_encoder.classes_,
            "num_classes": len(label_encoder.classes_),
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "test_samples": len(test_records),
        },
        "train": {
            "prompt_ids": [r["prompt_id"] for r in train_records],
            "X": X_train_seq,
            "y": y_train,
            "y_one_hot": y_train_one_hot,
            "scores": [float(r["overall_score"]) for r in train_records],
        },
        "val": {
            "prompt_ids": [r["prompt_id"] for r in val_records],
            "X": X_val_seq,
            "y": y_val,
            "y_one_hot": y_val_one_hot,
            "scores": [float(r["overall_score"]) for r in val_records],
        },
        "test": {
            "prompt_ids": [r["prompt_id"] for r in test_records],
            "X": X_test_seq,
            "y": y_test,
            "y_one_hot": y_test_one_hot,
            "scores": [float(r["overall_score"]) for r in test_records],
        },
    }

    with open(SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits_payload, f)

    print(f"[+] Step 6: Saved Numerical Tensor Splits to: {SPLITS_JSON}")

    # Step 10: Print Verification Sample
    print("\n" + "-" * 75)
    print("PREPROCESSING VERIFICATION CHECKPOINT:")
    print("-" * 75)
    sample_text = train_records[0]["prompt"]
    sample_clean = train_records[0]["clean_prompt"]
    sample_seq = X_train_seq[0]
    sample_label = train_records[0]["quality_label"]
    sample_y = y_train[0]
    sample_one_hot = y_train_one_hot[0]

    print(f"Original Text  : \"{sample_text[:80]}...\"")
    print(f"Cleaned Text   : \"{sample_clean[:80]}...\"")
    print(f"Target Label   : {sample_label} -> Encoded Index: {sample_y} | One-Hot: {sample_one_hot}")
    print(f"Padded Sequence: {sample_seq[:15]}... (Total {len(sample_seq)} tokens)")
    print("-" * 75)
    print("[SUCCESS] Phase 2 NLP Preprocessing Pipeline Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_preprocessing_pipeline()
