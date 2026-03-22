class Config:
    # ── Vision encoder ─────────────────────────────────────────────
    img_size      = 224
    patch_size    = 16
    vision_dim    = 768
    vision_depth  = 12
    vision_heads  = 12

    # ── Text encoder ───────────────────────────────────────────────
    vocab_size    = 8192   # BPE vocabulary size
    num_merges    = 4000   # BPE merge operations
    text_dim      = 512
    text_depth    = 12
    text_heads    = 8
    pad_token_id  = 0      # BPETokenizer.PAD_ID

    # ── Shared projection space ────────────────────────────────────
    proj_dim      = 512
    dropout       = 0.0

    # ── Sequence length ────────────────────────────────────────────
    max_seq_length = 64

    # ── Training ───────────────────────────────────────────────────
    batch_size    = 16
    epochs        = 10
    lr            = 5e-4    # higher LR — training from scratch
    weight_decay  = 0.05
    warmup_steps  = 500

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