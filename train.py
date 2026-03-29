import torch
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataloader import get_dataloaders, TOKENIZER_PATH
from tokenizer import BPETokenizer
from model import SigLIP2Model


# ─────────────────────────────────────────────
#  LR schedule
# ─────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
#  Image helpers  (zero torchvision dependency)
# ─────────────────────────────────────────────

_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
_STD  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def _denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised float → (H,W,3) uint8 numpy array."""
    t = (tensor.cpu() * _STD + _MEAN).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _pil_to_chw(img: Image.Image) -> np.ndarray:
    """PIL RGB → (3, H, W) uint8 numpy — format TensorBoard expects."""
    return np.array(img).transpose(2, 0, 1)          # HWC → CHW


def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap(text: str, max_chars: int) -> list:
    """Word-wrap text into lines of at most max_chars."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            if line:
                lines.append(line.strip())
            line = w + " "
        else:
            line += w + " "
    if line.strip():
        lines.append(line.strip())
    return lines


# ─────────────────────────────────────────────
#  Prediction panel builder
# ─────────────────────────────────────────────

PANEL_H   = 260          # total canvas height
IMG_W     = 224          # image square size
IMG_H     = 224
TEXT_W    = 380          # right-side text panel width
CANVAS_W  = IMG_W + TEXT_W


def _build_panel(
    img_arr:  np.ndarray,   # (H, W, 3) uint8
    gt:       str,          # ground-truth caption
    ranked:   list,         # [(caption_str, prob_float), ...]  top-N
) -> np.ndarray:
    """Return a (3, PANEL_H, CANVAS_W) uint8 numpy array."""

    canvas = Image.new("RGB", (CANVAS_W, PANEL_H), (20, 20, 20))

    # ── Paste image on the left ────────────────────────────────────
    pil_img = Image.fromarray(img_arr).resize((IMG_W, IMG_H), Image.BICUBIC)
    canvas.paste(pil_img, (0, (PANEL_H - IMG_H) // 2))

    draw      = ImageDraw.Draw(canvas)
    bold      = _try_font(13)
    reg       = _try_font(12)
    small     = _try_font(11)

    x0 = IMG_W + 10
    y  = 8

    # ── Ground-truth caption ──────────────────────────────────────
    draw.text((x0, y), "▶ Ground truth:", font=bold, fill=(255, 215, 0))
    y += 17
    for line in _wrap(gt, 42):
        draw.text((x0 + 4, y), line, font=reg, fill=(210, 210, 210))
        y += 15

    y += 4
    draw.line([(x0, y), (CANVAS_W - 6, y)], fill=(70, 70, 70), width=1)
    y += 8

    # ── Ranked predictions ────────────────────────────────────────
    draw.text((x0, y), "▶ Model scores (this batch):", font=bold, fill=(255, 215, 0))
    y += 17

    bar_max = TEXT_W - 60          # max bar pixel width
    for rank, (caption, prob) in enumerate(ranked):
        if y > PANEL_H - 24:
            break

        # Highlight correct match in green, rest in grey
        is_correct = (caption.strip() == gt.strip())
        bar_colour  = (80,  200, 80)  if is_correct else (80,  130, 200)
        txt_colour  = (120, 255, 120) if is_correct else (180, 180, 180)
        marker      = "✓" if is_correct else f"{rank+1}."

        label = caption if len(caption) <= 40 else caption[:37] + "…"
        draw.text((x0, y), f"{marker} {label}", font=small, fill=txt_colour)
        y += 13

        # Probability bar
        bar_w = max(2, int(prob * bar_max))
        draw.rectangle([x0, y, x0 + bar_w,     y + 8], fill=bar_colour)
        draw.rectangle([x0 + bar_w, y, x0 + bar_max, y + 8], fill=(45, 45, 45))
        draw.text((x0 + bar_max + 4, y), f"{prob:.3f}", font=small, fill=txt_colour)
        y += 14

    return _pil_to_chw(canvas)           # (3, H, W) uint8


# ─────────────────────────────────────────────
#  Visualise & log to TensorBoard
# ─────────────────────────────────────────────

def log_predictions(
    writer:      SummaryWriter,
    model:       SigLIP2Model,
    batch:       dict,
    tokenizer:   BPETokenizer,
    cfg,
    epoch:       int,
    tag_prefix:  str = "Predictions",
    n:           int = 2,
):
    """
    For each of the first `n` images in `batch`:
      - run the model against all captions in the batch
      - draw a panel: image | ground-truth | ranked scores
      - push it to TensorBoard under  <tag_prefix>/sample_0 … sample_n
    """
    model.eval()
    with torch.no_grad():
        pv  = batch["pixel_values"].to(cfg.device)
        ids = batch["input_ids"].to(cfg.device)
        am  = batch["attention_mask"].to(cfg.device)

        with torch.amp.autocast('cuda'):
            out = model(pixel_values=pv, input_ids=ids,
                        attention_mask=am, return_loss=False)

        probs = torch.sigmoid(out.logits_per_image).cpu()   # (B, B)

    n = min(n, pv.size(0))

    # Decode every caption once
    all_captions = [
        tokenizer.decode(ids[j].cpu().tolist())
        for j in range(ids.size(0))
    ]

    for i in range(n):
        gt       = all_captions[i]
        img_arr  = _denorm_to_uint8(pv[i])                 # (H,W,3)

        # Rank all captions by this image's similarity scores
        ranked = sorted(
            zip(all_captions, probs[i].tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:6]

        panel = _build_panel(img_arr, gt, ranked)           # (3,H,W) uint8

        writer.add_image(
            tag   = f"{tag_prefix}/sample_{i}",
            img_tensor = panel,
            global_step = epoch,
            dataformats = "CHW",
        )

    writer.flush()          # ← force write to disk so TensorBoard sees it
    model.train()


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────

def train():
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    print("Building model from scratch...")
    model = SigLIP2Model(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    train_loader, val_loader = get_dataloaders(cfg)
    total_steps = len(train_loader) * cfg.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda')
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    # Fixed validation batch — same samples every epoch so you can
    # watch the model improve in the IMAGES tab across epochs
    fixed_val_batch = next(iter(val_loader))

    best_val_loss = float("inf")
    global_step   = 0
    VAL_FREQ      = 500   # ← validate every this many steps

    def run_validation(step: int):
        """Full val-set pass + image panels logged at `step`. Saves best ckpt."""
        nonlocal best_val_loss
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for vbatch in val_loader:
                vpv  = vbatch["pixel_values"].to(cfg.device)
                vids = vbatch["input_ids"].to(cfg.device)
                vam  = vbatch["attention_mask"].to(cfg.device)
                with torch.amp.autocast('cuda'):
                    vout = model(pixel_values=vpv, input_ids=vids,
                                 attention_mask=vam, return_loss=True)
                v_loss += vout.loss.item()

        avg_v = v_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_v, step)

        # Fixed val prediction panels in IMAGES tab
        log_predictions(writer, model, fixed_val_batch, tokenizer, cfg,
                        epoch=step, tag_prefix="Predictions/val", n=2)
        writer.flush()

        print(f"\n  [step {step}] val_loss={avg_v:.4f}")

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            model.save(cfg.checkpoint_best)
            print(f"  ⭐ Best model → {cfg.checkpoint_best}  (val={avg_v:.4f})")

        model.train()
        return avg_v

    for epoch in range(cfg.epochs):
        # ── Training ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        last_batch = None
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]")

        for batch in pbar:
            pv  = batch["pixel_values"].to(cfg.device)
            ids = batch["input_ids"].to(cfg.device)
            am  = batch["attention_mask"].to(cfg.device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out  = model(pixel_values=pv, input_ids=ids,
                             attention_mask=am, return_loss=True)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_lr  = scheduler.get_last_lr()[0]
            train_loss += loss.item()
            global_step += 1

            writer.add_scalar("Loss/train_step",   loss.item(),  global_step)
            writer.add_scalar("LearningRate/step", current_lr,   global_step)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
            last_batch = batch

            # ── Validate every VAL_FREQ steps ─────────────────────
            if global_step % VAL_FREQ == 0:
                run_validation(step=global_step)

        avg_train  = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Loss/train_epoch",   avg_train,  epoch)
        writer.add_scalar("LearningRate/epoch", current_lr, epoch)

        print(
            f"Epoch {epoch+1}/{cfg.epochs}  "
            f"train={avg_train:.4f}  lr={current_lr:.2e}"
        )

        # Train prediction panels at end of each epoch
        if last_batch is not None:
            log_predictions(writer, model, last_batch, tokenizer, cfg,
                            epoch=global_step, tag_prefix="Predictions/train", n=2)
            writer.flush()

    model.save(cfg.checkpoint_final)
    print(f"\nDone. Final model → {cfg.checkpoint_final}")
    writer.close()


if __name__ == "__main__":
    train()