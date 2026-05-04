"""
Phase 3 — TrOCR Fine-Tuning Trainer
Handwritten Document Reader
═══════════════════════════════════════════════════════════════════════════════
PURPOSE
  Fine-tunes Microsoft's TrOCR model on the synthetic Taglish line image
  dataset generated in Phase 2. After training, the model can read
  handwritten English, Tagalog, and Taglish text from line images.

WHAT IS TrOCR?
  TrOCR (Transformer-based Optical Character Recognition) is a model
  published by Microsoft Research in 2021. It combines:
    - A Vision Transformer (ViT) encoder — reads the image
    - A RoBERTa-based decoder — generates the text sequence

  We start from the pretrained checkpoint:
    microsoft/trocr-base-handwritten
  This checkpoint was already trained on English handwriting datasets
  (IAM, IMGUR5K). We fine-tune it further on our Taglish corpus so it
  learns Tagalog words, Taglish mixing, and common abbreviations.

  Fine-tuning is orders of magnitude faster than training from scratch.
  Instead of learning what letters look like (already done), the model
  only needs to learn the Taglish-specific vocabulary and patterns.

WHY TrOCR OVER CRNN?
  CRNN+CTC was the industry standard until ~2021. TrOCR surpasses it
  because the attention mechanism allows it to handle:
    - Variable-length output without explicit segmentation
    - Long-range dependencies (a letter at position 10 can inform
      the prediction at position 1 during beam search)
    - Better handling of ambiguous characters in context

TRAINING STRATEGY — TWO STAGES
  Stage 1 (Epochs 1-5): Freeze the encoder, train decoder only
    → Adapts the language model to Taglish vocabulary quickly
    → Lower risk of catastrophic forgetting of visual features
    → Faster convergence in the first few epochs

  Stage 2 (Epochs 6-10): Unfreeze full model, lower learning rate
    → Fine-tunes visual features for Taglish handwriting style
    → Allows encoder to specialize for our font/augmentation distribution

EVALUATION METRIC — CHARACTER ERROR RATE (CER)
  CER = (Substitutions + Insertions + Deletions) / Total Characters
  Target: CER < 0.10 (less than 10% character-level errors)
  Why CER and not accuracy? Because "Kamusta" vs "Kamusta." differs
  by one character — accuracy would call this wrong, CER would give
  partial credit. CER is the standard metric for HTR systems.

OUTPUT
  checkpoints/
    best_model/          ← best checkpoint by val CER (use this for inference)
    checkpoint_epoch_N/  ← checkpoint per epoch
  logs/
    training_log.json    ← full training history for W&B or analysis
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Paths
LABELS_CSV      = Path("data/taglish_lines/labels.csv")
IMAGES_DIR      = Path("data/taglish_lines")
CHECKPOINTS_DIR = Path("checkpoints")
LOGS_DIR        = Path("logs")

# Model
MODEL_NAME      = "microsoft/trocr-base-handwritten"

# Training — Stage 1 (decoder only)
STAGE1_EPOCHS   = 5
STAGE1_LR       = 5e-5

# Training — Stage 2 (full model)
STAGE2_EPOCHS   = 5
STAGE2_LR       = 1e-5

# Data
BATCH_SIZE      = 16      # TrOCR is large — 16 fits safely on T4 GPU
MAX_TARGET_LEN  = 128     # max characters per line
NUM_WORKERS     = 2
RANDOM_SEED     = 42

# Early stopping
PATIENCE        = 3       # stop if val CER doesn't improve for 3 epochs

# Target
TARGET_CER      = 0.10    # 10% — move to Phase 4 when this is reached


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = RANDOM_SEED):
    """
    Set all random seeds for reproducibility.

    WHY SET MULTIPLE SEEDS?
      Python, NumPy, and PyTorch each maintain their own random state.
      Setting only one does not affect the others. All three must be
      seeded for fully reproducible training runs.

    WHY torch.backends.cudnn.deterministic = True?
      Some CUDA operations are non-deterministic by default for speed.
      Setting deterministic=True forces reproducible GPU operations
      at a small performance cost. Worth it during development.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TaqlishLineDataset(Dataset):
    """
    PyTorch Dataset for Taglish handwritten line images.

    WHAT IT DOES:
      Loads image paths and labels from the Phase 2 labels.csv.
      On __getitem__, opens the image and returns it alongside
      its text label. The TrOCR processor handles all image
      preprocessing (resize, normalize) internally.

    WHY NOT PRE-LOAD ALL IMAGES INTO RAM?
      33,240 images × ~32KB each ≈ 1GB RAM just for images.
      Colab's T4 instance has 12GB RAM total. Pre-loading would
      leave little room for the model and gradients.
      Lazy loading (open on demand) uses near-zero RAM at rest.

    WHY USE expanded_text AS THE LABEL?
      We train the model to output expanded text rather than
      abbreviated text. This means the model learns to both
      READ the handwriting AND UNDERSTAND abbreviations in one step.
      A model trained on 'text' would output "pls submit asap".
      A model trained on 'expanded_text' outputs "please submit
      as soon as possible" — more useful for downstream tasks.

    NOTE ON __len__ AND __getitem__:
      These two methods are the only requirement for a PyTorch Dataset.
      __len__ tells the DataLoader how many samples exist.
      __getitem__ returns one sample by index.
      Everything else (batching, shuffling, parallelism) is handled
      automatically by DataLoader.
    """

    def __init__(self, df: pd.DataFrame, images_dir: Path,
                 processor: TrOCRProcessor, max_target_len: int = MAX_TARGET_LEN):
        self.df           = df.reset_index(drop=True)
        self.images_dir   = images_dir
        self.processor    = processor
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load image
        img_path = self.images_dir / row["filename"]
        img = Image.open(img_path).convert("RGB")
        # TrOCR processor expects RGB — even though our images are grayscale,
        # converting to RGB replicates the single channel across R, G, B.
        # The processor then handles its own normalization internally.

        # Process image — resize to 384×384, normalize to ImageNet stats
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        ).pixel_values.squeeze(0)
        # squeeze(0) removes the batch dimension added by the processor
        # shape goes from (1, 3, 384, 384) to (3, 384, 384)

        # Tokenize label
        label = str(row["expanded_text"])
        tokenized = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)

        # Replace padding token id with -100
        # WHY -100? PyTorch's CrossEntropyLoss ignores index -100 by default.
        # Padding positions should not contribute to the loss — the model
        # should not be penalized for not predicting padding tokens.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels":       labels,
        }


def build_dataloaders(labels_csv: Path, images_dir: Path,
                      processor: TrOCRProcessor,
                      batch_size: int = BATCH_SIZE,
                      num_workers: int = NUM_WORKERS) -> tuple:
    """
    Load Phase 2 labels.csv and build train/val/test DataLoaders.

    WHY READ FROM labels.csv INSTEAD OF RE-SPLITTING?
      Phase 2 already performed the split and recorded it in labels.csv.
      Re-splitting here would produce different splits, breaking the
      contract between phases. We respect Phase 2's split decisions.

    WHY pin_memory=True?
      Pinned memory allows faster CPU→GPU transfer. On a T4 GPU this
      can improve data loading throughput by 20-40%.
    """
    df = pd.read_csv(labels_csv, encoding="utf-8")
    print(f"  Loaded {len(df):,} records from labels.csv")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"  Split → train: {len(train_df):,} | "
          f"val: {len(val_df):,} | test: {len(test_df):,}")

    train_ds = TaqlishLineDataset(train_df, images_dir, processor)
    val_ds   = TaqlishLineDataset(val_df,   images_dir, processor)
    test_ds  = TaqlishLineDataset(test_df,  images_dir, processor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cer(predictions: list, references: list) -> float:
    """
    Compute Character Error Rate (CER) using edit distance.

    CER = (S + I + D) / N
    where:
      S = substitutions (wrong character)
      I = insertions    (extra character)
      D = deletions     (missing character)
      N = total characters in reference

    WHY EDIT DISTANCE?
      Edit distance (Levenshtein distance) counts the minimum number
      of single-character edits to transform prediction into reference.
      This directly maps to S+I+D in the CER formula.

    WHY NOT WORD ERROR RATE (WER)?
      WER treats each word as atomic — "Kamusta" vs "Kamust" is one
      full word error. CER gives partial credit for partially correct
      words. For handwriting recognition, CER is more informative
      because single-character OCR errors are common and expected.

    IMPLEMENTATION NOTE:
      We implement edit distance with dynamic programming (Wagner-Fischer
      algorithm). Time complexity O(m×n) where m,n are string lengths.
      This is the standard implementation used in the HTR research community.
    """
    total_chars  = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref  = ref.strip()

        # Wagner-Fischer edit distance
        m, n = len(ref), len(pred)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == pred[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )

        total_errors += dp[m][n]
        total_chars  += max(m, 1)  # avoid division by zero

    return total_errors / total_chars


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler,
                    device, epoch: int) -> float:
    """
    Run one full training epoch.

    WHAT IT DOES:
      Iterates all batches in the training DataLoader.
      For each batch:
        1. Forward pass — model computes predictions and loss
        2. Backward pass — compute gradients
        3. Gradient clipping — prevent exploding gradients
        4. Optimizer step — update weights
        5. Scheduler step — adjust learning rate
      Returns average loss for the epoch.

    WHY GRADIENT CLIPPING (max_norm=1.0)?
      Transformers can produce very large gradients early in fine-tuning,
      especially when unfreezing the encoder in Stage 2. Large gradients
      cause weight updates that are too large, destabilizing training.
      Clipping gradients to a maximum norm of 1.0 prevents this.
      This is standard practice for all transformer fine-tuning.

    WHY model.train()?
      PyTorch models have two modes: train and eval.
      In train mode: dropout is active, batch norm uses batch statistics.
      In eval mode: dropout is disabled, batch norm uses running statistics.
      You must explicitly switch modes — forgetting model.train() before
      training is a common bug that causes mysteriously poor performance.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"    Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        # Forward pass
        # VisionEncoderDecoderModel computes cross-entropy loss internally
        # when labels are provided — no need to compute it manually
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss    = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Weight update
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, processor, device,
             split_name: str = "val") -> tuple:
    """
    Evaluate model on a DataLoader. Returns (avg_loss, CER).

    WHAT IT DOES:
      Runs inference on all batches without computing gradients.
      Decodes model output tokens back to text strings.
      Computes CER against ground truth labels.

    WHY @torch.no_grad()?
      During evaluation, we don't need to compute gradients — we're not
      updating weights. Disabling gradient computation saves ~30% memory
      and speeds up inference significantly. Always use this decorator
      for evaluation and inference loops.

    WHY skip_special_tokens=True?
      The tokenizer adds special tokens: [BOS] (beginning of sequence),
      [EOS] (end of sequence), [PAD] (padding). These are internal to the
      model and should never appear in the output text. Skipping them
      gives clean human-readable predictions.

    WHY RECONSTRUCT LABELS FROM -100?
      We set padding positions to -100 during dataset creation.
      To decode the original label text, we need to replace -100 back
      with the pad token ID before calling the tokenizer decoder.
    """
    model.eval()
    total_loss   = 0.0
    all_preds    = []
    all_refs     = []

    pbar = tqdm(loader, desc=f"    [{split_name}]", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        # Compute loss
        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

        # Generate predictions (greedy decoding)
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=MAX_TARGET_LEN,
        )

        # Decode predictions to text
        preds = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Decode labels — replace -100 back to pad_token_id first
        label_ids = labels.clone()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(
            label_ids, skip_special_tokens=True
        )

        all_preds.extend(preds)
        all_refs.extend(refs)

    avg_loss = total_loss / len(loader)
    cer      = compute_cer(all_preds, all_refs)

    # Show 3 example predictions so you can see qualitative progress
    print(f"\n    Sample predictions [{split_name}]:")
    for pred, ref in zip(all_preds[:3], all_refs[:3]):
        print(f"      REF : {ref}")
        print(f"      PRED: {pred}")
        print()

    return avg_loss, cer


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str = MODEL_NAME, device: torch.device = None):
    """
    Load TrOCR processor and model from HuggingFace.

    WHAT IS THE PROCESSOR?
      TrOCRProcessor combines two components:
        1. ViTImageProcessor — preprocesses images (resize, normalize)
        2. RobertaTokenizer  — tokenizes text labels

      The processor is the interface between raw data (PIL images, strings)
      and model-ready tensors. Always use the processor that matches the
      model checkpoint — they must be from the same release.

    WHAT IS VisionEncoderDecoderModel?
      A HuggingFace wrapper that connects:
        - Encoder: ViT (Vision Transformer) — reads the image
        - Decoder: RoBERTa — generates the text token by token
      The model internally handles cross-attention between visual
      features and generated tokens.

    WHY SET decoder_start_token_id AND eos_token_id?
      These tell the model:
        - Where to start generating (BOS token)
        - When to stop generating (EOS token)
      Without these, generation would either never start or never stop.
    """
    print(f"  Loading processor from {model_name}...")
    processor = TrOCRProcessor.from_pretrained(model_name)

    print(f"  Loading model from {model_name}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Required generation config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    # Generation settings
    model.config.max_new_tokens  = MAX_TARGET_LEN
    model.config.early_stopping  = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty  = 2.0
    model.config.num_beams       = 4
    # num_beams=4: beam search considers 4 candidate sequences at each step
    # More beams = better output quality but slower inference
    # 4 is the standard trade-off for HTR tasks

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded — {total_params:,} parameters")
    print(f"  Device: {device}")

    return processor, model


def freeze_encoder(model):
    """
    Freeze all encoder parameters for Stage 1 training.

    WHY FREEZE THE ENCODER IN STAGE 1?
      The pretrained encoder already knows how to extract visual features
      from handwriting — it was trained on millions of images. If we unfreeze
      it immediately, the large gradients from the Taglish vocabulary adaptation
      could corrupt these visual features (catastrophic forgetting).

      By freezing the encoder first, we let the decoder adapt to Taglish
      vocabulary while the encoder stays stable. Only in Stage 2 do we
      unfreeze the encoder for fine-grained visual adaptation.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False

    frozen   = sum(p.numel() for p in model.encoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Stage 1: Encoder frozen ({frozen:,} params frozen, "
          f"{trainable:,} trainable)")


def unfreeze_encoder(model):
    """Unfreeze all parameters for Stage 2 full fine-tuning."""
    for param in model.encoder.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Stage 2: Full model unfrozen ({trainable:,} trainable params)")


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, processor, epoch: int, cer: float,
                    checkpoints_dir: Path, is_best: bool = False):
    """
    Save model checkpoint after each epoch.

    WHAT IT SAVES:
      - model weights and config (via HuggingFace save_pretrained)
      - processor (tokenizer + image processor)
      - epoch number and CER for reference

    WHY save_pretrained INSTEAD OF torch.save?
      HuggingFace's save_pretrained saves in a format that can be loaded
      with from_pretrained — including full config, tokenizer vocab, and
      special token mappings. torch.save only saves weights — you'd need
      to manually reconstruct the config, making loading fragile.

    WHY SAVE EVERY EPOCH + BEST SEPARATELY?
      Per-epoch checkpoints let you roll back if something goes wrong.
      The best checkpoint is what you deploy — it's the one with the
      lowest validation CER across all epochs.
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save per-epoch checkpoint
    epoch_dir = checkpoints_dir / f"checkpoint_epoch_{epoch:02d}_cer_{cer:.4f}"
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    # Save best checkpoint separately
    if is_best:
        best_dir = checkpoints_dir / "best_model"
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)

        # Save metadata
        meta = {"epoch": epoch, "val_cer": cer,
                "saved_at": datetime.utcnow().isoformat() + "Z"}
        with open(best_dir / "best_model_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ★ New best model saved (epoch {epoch}, CER={cer:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOG
# ══════════════════════════════════════════════════════════════════════════════

def save_log(log: list, logs_dir: Path):
    """Save full training history as JSON for analysis."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "training_log.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train(
    labels_csv      = LABELS_CSV,
    images_dir      = IMAGES_DIR,
    checkpoints_dir = CHECKPOINTS_DIR,
    logs_dir        = LOGS_DIR,
    model_name      = MODEL_NAME,
    stage1_epochs   = STAGE1_EPOCHS,
    stage2_epochs   = STAGE2_EPOCHS,
    stage1_lr       = STAGE1_LR,
    stage2_lr       = STAGE2_LR,
    batch_size      = BATCH_SIZE,
    patience        = PATIENCE,
):
    """
    Full Phase 3 training pipeline.

    RETURNS: (model, processor, training_log)
    """
    set_seed(RANDOM_SEED)

    labels_csv      = Path(labels_csv)
    images_dir      = Path(images_dir)
    checkpoints_dir = Path(checkpoints_dir)
    logs_dir        = Path(logs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "═" * 60)
    print("  Phase 3 — TrOCR Fine-Tuning")
    print("  Handwritten Document Reader")
    print("═" * 60)
    print(f"\n  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading TrOCR model and processor...")
    processor, model = load_model(model_name, device)

    # ── Build dataloaders ─────────────────────────────────────────────────────
    print("\n[2/5] Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        labels_csv, images_dir, processor, batch_size
    )

    training_log = []
    best_cer     = float("inf")
    no_improve   = 0
    total_epochs = stage1_epochs + stage2_epochs

    # ── STAGE 1: Decoder-only training ───────────────────────────────────────
    print(f"\n[3/5] Stage 1 — Decoder-only training ({stage1_epochs} epochs)...")
    freeze_encoder(model)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * stage1_epochs
    warmup_steps = total_steps // 10  # 10% warmup

    # WHY WARMUP?
    # At the start of training, weights are in a random state relative
    # to our task. A large learning rate immediately can cause instability.
    # Warmup gradually increases LR from 0 to stage1_lr over the first
    # 10% of steps, giving the optimizer time to find a stable direction.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(1, stage1_epochs + 1):
        print(f"\n  Epoch {epoch}/{total_epochs} [Stage 1 — decoder only]")
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_cer = evaluate(model, val_loader, processor, device, "val")

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_CER={val_cer:.4f} | "
              f"time={elapsed:.0f}s")

        is_best = val_cer < best_cer
        if is_best:
            best_cer   = val_cer
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(model, processor, epoch, val_cer,
                        checkpoints_dir, is_best)

        log_entry = {
            "epoch": epoch, "stage": 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_cer,
            "best_cer": best_cer,
            "elapsed_s": elapsed,
        }
        training_log.append(log_entry)
        save_log(training_log, logs_dir)

        # Early stopping check
        if no_improve >= patience:
            print(f"\n  Early stopping — no improvement for {patience} epochs")
            break

        if val_cer <= TARGET_CER:
            print(f"\n  Target CER {TARGET_CER} reached at epoch {epoch}!")
            break

    # ── STAGE 2: Full model fine-tuning ──────────────────────────────────────
    print(f"\n[4/5] Stage 2 — Full model fine-tuning ({stage2_epochs} epochs)...")
    unfreeze_encoder(model)
    no_improve = 0  # reset patience counter for stage 2

    optimizer = AdamW(
        model.parameters(),
        lr=stage2_lr,      # lower LR — encoder is sensitive, don't overwrite
        weight_decay=0.01,
    )
    total_steps_s2 = len(train_loader) * stage2_epochs
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps_s2, eta_min=1e-6
    )
    # WHY COSINE ANNEALING IN STAGE 2?
    # CosineAnnealingLR smoothly reduces LR following a cosine curve.
    # By Stage 2, the decoder is already adapted. The encoder needs gentle,
    # decreasing updates to fine-tune without overwriting pretrained features.
    # Cosine decay is smoother than linear — better for fine-grained tuning.

    for epoch in range(stage1_epochs + 1, total_epochs + 1):
        print(f"\n  Epoch {epoch}/{total_epochs} [Stage 2 — full model]")
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_cer = evaluate(model, val_loader, processor, device, "val")

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_CER={val_cer:.4f} | "
              f"time={elapsed:.0f}s")

        is_best = val_cer < best_cer
        if is_best:
            best_cer   = val_cer
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(model, processor, epoch, val_cer,
                        checkpoints_dir, is_best)

        log_entry = {
            "epoch": epoch, "stage": 2,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_cer,
            "best_cer": best_cer,
            "elapsed_s": elapsed,
        }
        training_log.append(log_entry)
        save_log(training_log, logs_dir)

        if no_improve >= patience:
            print(f"\n  Early stopping — no improvement for {patience} epochs")
            break

        if val_cer <= TARGET_CER:
            print(f"\n  Target CER {TARGET_CER} reached at epoch {epoch}!")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n[5/5] Final evaluation on test set...")
    print("  Loading best model checkpoint...")
    best_model = VisionEncoderDecoderModel.from_pretrained(
        checkpoints_dir / "best_model"
    ).to(device)

    _, test_cer = evaluate(best_model, test_loader, processor, device, "test")

    print("\n" + "═" * 60)
    print("  Phase 3 Complete")
    print(f"  Best val CER  : {best_cer:.4f} ({best_cer*100:.1f}%)")
    print(f"  Test CER      : {test_cer:.4f} ({test_cer*100:.1f}%)")
    print(f"  Target CER    : {TARGET_CER:.4f} ({TARGET_CER*100:.1f}%)")
    status = "✅ PASSED" if test_cer <= TARGET_CER else "⚠️  not yet — consider more data or epochs"
    print(f"  Status        : {status}")
    print(f"  Best model    : {checkpoints_dir / 'best_model'}")
    print("═" * 60 + "\n")

    return model, processor, training_log


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, processor, log = train()
