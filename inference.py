"""
Inference — fully from scratch, no transformers dependency.
"""

import torch
from PIL import Image
from config import Config
from model import SigLIP2Model
from tokenizer import BPETokenizer
from image_processor import ImageProcessor

TOKENIZER_PATH = "bpe_tokenizer.json"


def run_inference(image_path, labels, model_weights_path=None):
    cfg = Config()
    if model_weights_path is None:
        model_weights_path = cfg.checkpoint_final

    # Load tokenizer trained during training
    tokenizer       = BPETokenizer.load(TOKENIZER_PATH)
    image_processor = ImageProcessor(img_size=cfg.img_size)

    # Load model
    model = SigLIP2Model.load(model_weights_path, cfg, device=cfg.device)
    model.eval()

    # Pre-process image
    image        = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image).unsqueeze(0).to(cfg.device)  # (1, 3, H, W)

    # Encode each label separately, stack into a batch
    input_ids_list = []
    attn_mask_list = []
    for label in labels:
        enc = tokenizer.encode(label, max_length=cfg.max_seq_length)
        input_ids_list.append(enc["input_ids"])
        attn_mask_list.append(enc["attention_mask"])

    input_ids      = torch.tensor(input_ids_list,  dtype=torch.long).to(cfg.device)
    attention_mask = torch.tensor(attn_mask_list,  dtype=torch.long).to(cfg.device)

    # Expand image to match number of labels
    pixel_values = pixel_values.expand(len(labels), -1, -1, -1)

    with torch.no_grad():
        outputs = model(
            pixel_values   = pixel_values,
            input_ids      = input_ids,
            attention_mask = attention_mask,
            return_loss    = False,
        )

    # Diagonal of (N, N) logit matrix = score for each (image_i, text_i) pair
    probs = torch.sigmoid(outputs.logits_per_image.diag())

    print("\n--- Inference Results ---")
    for label, prob in zip(labels, probs):
        print(f"  {label:35s} | p = {prob.item():.4f}")

    return dict(zip(labels, probs.tolist()))


if __name__ == "__main__":
    test_labels = [
        "a cat sitting on a fence",
        "a dog running in grass",
        "a person hiking in the mountains",
        "a crowded city street",
    ]
    run_inference("test_image.jpg", test_labels)