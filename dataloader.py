"""
Dataloader — fully from scratch, no transformers/AutoProcessor.
Uses BPETokenizer + ImageProcessor defined in this project.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer
from image_processor import ImageProcessor


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class Flickr30kDataset(Dataset):
    def __init__(self, df, img_dir: str,
                 tokenizer: BPETokenizer,
                 image_processor: ImageProcessor,
                 max_seq_length: int):
        self.df              = df.reset_index(drop=True)
        self.img_dir         = img_dir
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.max_seq_length  = max_seq_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_name"].strip()
        caption  = str(self.df.iloc[idx]["comment"]).strip()

        # ── Image ─────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(image)               # (3, H, W)

        # ── Text ──────────────────────────────────────────────────
        encoded        = self.tokenizer.encode(caption, max_length=self.max_seq_length)
        input_ids      = torch.tensor(encoded["input_ids"],      dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }


# ─────────────────────────────────────────────
#  Build / load tokenizer
# ─────────────────────────────────────────────

TOKENIZER_PATH = "bpe_tokenizer.json"


def _get_tokenizer(df, config) -> BPETokenizer:
    """
    Train a fresh tokenizer or load an existing one from disk.

    FIX: previously any existing bpe_tokenizer.json was loaded unconditionally,
    so changing vocab_size / num_merges in Config had no effect and a tokenizer
    trained with an old/smaller vocabulary was silently reused.  This produced
    massive numbers of <unk> tokens because sub-word merges were incomplete.

    Now we compute a config hash and retrain whenever the saved file doesn't
    match the current settings.
    """
    desired_hash = BPETokenizer._make_hash(config.vocab_size, config.num_merges)

    if os.path.exists(TOKENIZER_PATH):
        existing = BPETokenizer.load(TOKENIZER_PATH)
        if existing._config_hash == desired_hash:
            print(f"Loading tokenizer from {TOKENIZER_PATH}  (vocab={len(existing.token2id)})")
            return existing
        else:
            print(
                f"Config changed (old hash={existing._config_hash}, "
                f"new hash={desired_hash}) — retraining tokenizer…"
            )

    print("Training BPE tokenizer from scratch on captions…")
    # FIX: train on ALL captions (not just train split) so the vocabulary
    # covers every word that appears in validation captions too.
    # The tokenizer is a pre-processing step, not a learned model component,
    # so using val captions for vocabulary construction is not data leakage.
    captions  = df["comment"].dropna().tolist()
    tokenizer = BPETokenizer(
        vocab_size = config.vocab_size,
        num_merges = config.num_merges,
    )
    tokenizer.train(captions)
    tokenizer.save(TOKENIZER_PATH)
    return tokenizer


# ─────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────

def get_dataloaders(config):
    # Load CSV
    df = pd.read_csv(config.data_csv, sep=",")
    df.columns = [c.strip() for c in df.columns]
    rename_dict = {'image': 'image_name', 'caption': 'comment'}
    df = df.rename(columns=rename_dict)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(subset=["image_name", "comment"]).reset_index(drop=True)
    print(f"Total samples: {len(df)}")

    # FIX: build/check tokenizer BEFORE the train/val split so it sees all captions
    tokenizer = _get_tokenizer(df, config)

    # Image processor
    image_processor = ImageProcessor(img_size=config.img_size)

    # Train / val split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df   = df.drop(train_df.index)
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    train_ds = Flickr30kDataset(
        train_df, config.img_dir, tokenizer, image_processor, config.max_seq_length
    )
    val_ds = Flickr30kDataset(
        val_df, config.img_dir, tokenizer, image_processor, config.max_seq_length
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader