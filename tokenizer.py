"""
Simple BPE-style tokenizer built from scratch.
Trains a vocabulary from the dataset captions, then encodes/decodes text.
"""

import re
import json
import os
from collections import Counter, defaultdict


# ─────────────────────────────────────────────
#  Text cleaning
# ─────────────────────────────────────────────

def basic_clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 .,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_tokenize(text: str):
    """Split on spaces, keep punctuation attached via simple rules."""
    return text.split()


# ─────────────────────────────────────────────
#  BPE helpers
# ─────────────────────────────────────────────

def _get_pairs(vocab: dict):
    """Return a Counter of all adjacent symbol pairs across the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _merge_pair(pair, vocab: dict) -> dict:
    """Merge the most frequent pair in every word in the vocabulary."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab


# ─────────────────────────────────────────────
#  BPETokenizer
# ─────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch on a list of texts.

    Special tokens
    --------------
    <pad>  = 0
    <sos>  = 1
    <eos>  = 2
    <unk>  = 3
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_size: int = 8192, num_merges: int = 4000):
        self.vocab_size  = vocab_size
        self.num_merges  = num_merges
        self.merges      = []          # list of (pair_a, pair_b) tuples
        self.token2id    = {}
        self.id2token    = {}

    # ── Training ──────────────────────────────────────────────────

    def train(self, texts: list):
        """Train BPE on a list of raw strings."""
        print("Building character vocabulary…")

        # Start: every word → space-separated characters + end-of-word marker
        word_freq: Counter = Counter()
        for text in texts:
            for word in word_tokenize(basic_clean(text)):
                chars = " ".join(list(word)) + " </w>"
                word_freq[chars] += 1

        vocab = dict(word_freq)

        # Collect initial character-level tokens
        char_set = set()
        for word in vocab:
            char_set.update(word.split())

        # Build token2id: specials first, then chars, then merges
        special = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        all_tokens = special + sorted(char_set)
        self.token2id = {t: i for i, t in enumerate(all_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

        print(f"Starting vocab size: {len(self.token2id)}  |  running {self.num_merges} merges…")

        for step in range(self.num_merges):
            if len(self.token2id) >= self.vocab_size:
                break
            pairs = _get_pairs(vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            vocab = _merge_pair(best, vocab)
            self.merges.append(best)
            new_token = "".join(best)
            if new_token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[new_token] = idx
                self.id2token[idx] = new_token

            if (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{self.num_merges}  vocab={len(self.token2id)}")

        print(f"Training complete. Final vocab size: {len(self.token2id)}")

    # ── Encoding ──────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list:
        """Apply learned BPE merges to a single word."""
        symbols = list(word) + ["</w>"]
        symbols = " ".join(symbols)
        for pair in self.merges:
            bigram = " ".join(pair)
            replacement = "".join(pair)
            symbols = symbols.replace(bigram, replacement)
        return symbols.split()

    def encode(self, text: str, max_length: int = 64,
               add_special_tokens: bool = True) -> dict:
        """
        Returns:
            input_ids:      list[int]
            attention_mask: list[int]
        """
        tokens = []
        for word in word_tokenize(basic_clean(text)):
            tokens.extend(self._tokenize_word(word))

        ids = [self.token2id.get(t, self.UNK_ID) for t in tokens]

        if add_special_tokens:
            ids = [self.SOS_ID] + ids + [self.EOS_ID]

        # Truncate
        ids = ids[:max_length]

        # Pad
        pad_len = max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.PAD_ID] * pad_len

        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, ids: list) -> str:
        tokens = [self.id2token.get(i, self.UNK_TOKEN) for i in ids
                  if i not in (self.PAD_ID, self.SOS_ID, self.EOS_ID)]
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "num_merges": self.num_merges,
            "merges":     self.merges,
            "token2id":   self.token2id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"], num_merges=data["num_merges"])
        tok.merges   = [tuple(m) for m in data["merges"]]
        tok.token2id = data["token2id"]
        tok.id2token = {int(i): t for t, i in tok.token2id.items()}
        return tok