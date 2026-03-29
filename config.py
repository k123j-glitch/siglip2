class Config:
    # ── Vision encoder ─────────────────────────────────────────────
    img_size      = 224
    patch_size    = 16
    vision_dim    = 768
    vision_depth  = 12      # FIX: was 24 (ViT-L size), 12 is ViT-B — appropriate for scratch training
    vision_heads  = 12

    # ── Text encoder ───────────────────────────────────────────────
    vocab_size    = 16000  # FIX: was 8192 — larger vocab = fewer UNK tokens, better text representations
    num_merges    = 8000   # FIX: was 4000 — more merges = words represented as whole tokens not char fragments
    text_dim      = 512
    text_depth    = 6       # FIX: was 24 — text encoder can be shallower than vision
    text_heads    = 8
    pad_token_id  = 0      # BPETokenizer.PAD_ID

    # ── Shared projection space ────────────────────────────────────
    proj_dim      = 512
    dropout       = 0.1    # FIX: was 0.0 — small dropout helps regularise scratch training

    # ── Sequence length ────────────────────────────────────────────
    max_seq_length = 64

    # ── Training ───────────────────────────────────────────────────
    batch_size    = 64      # FIX: was 16 — contrastive loss needs many negatives; 64 minimum
    epochs        = 30      # FIX: was 10 — more epochs needed for scratch training
    lr            = 1e-4    # FIX: was 5e-4 — lower LR for deeper model, avoids loss spikes
    weight_decay  = 0.05
    warmup_steps  = 1000   # FIX: was 500 — more warmup for larger batch

    # ── Hardware ───────────────────────────────────────────────────
    device        = "cuda"
    num_workers   = 0

    # ── Paths ──────────────────────────────────────────────────────
    data_csv         = "data/flickr30k_images/captions.txt"
    img_dir          = "data/flickr30k_images/images"
    checkpoint_best  = "siglip2_best.pt"
    checkpoint_final = "siglip2_final.pt"

    # ── Logging ────────────────────────────────────────────────────
    log_dir      = "runs/siglip2"
    project_name = "siglip2-flickr-scratch"