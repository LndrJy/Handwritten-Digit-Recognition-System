"""
Phase 2 — Synthetic Taglish Line Image Dataset Generator
Handwritten Document Reader
═══════════════════════════════════════════════════════════════════════════════
PURPOSE
  Generates synthetic handwritten line images from a Taglish sentence corpus.
  Each sentence is rendered using multiple handwriting-style fonts, then
  augmented to simulate real-world variation. The output is a labeled dataset
  of line images consumable by TrOCR in Phase 3.

WHY LINE IMAGES?
  TrOCR (the model we use in Phase 3) reads one text line at a time.
  It was designed and trained this way. Feeding it a full document page
  directly would produce poor results. By generating line-level images now,
  our dataset perfectly matches the format TrOCR expects.

OUTPUT
  data/
    taglish_lines/
      images/
        train/   ← 80% of generated images
        val/     ← 10%
        test/    ← 10%
      labels.csv
      dataset_manifest_phase2.json
      sample_preview.png

EXPECTED SCALE
  ~279 sentences × 8 font variants × 5 augmentations = ~11,160 images
  Enough for initial fine-tuning. More can be added by expanding the corpus.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import csv
import json
import random
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CORPUS_PATH   = Path("data/sentences/taglish_corpus.txt")
ABBREV_PATH   = Path("data/sentences/abbreviations.json")
OUTPUT_DIR    = Path("data/taglish_lines")
FONTS_DIR     = Path("data/fonts")

# Image dimensions
# Height is fixed — TrOCR expects consistent line height
# Width is dynamic — padded/cropped to MAX_WIDTH
IMG_HEIGHT    = 64
MAX_WIDTH     = 512

# Training config
AUGMENT_FACTOR = 5       # augmented copies per sentence per font
RANDOM_SEED    = 42
VAL_SPLIT      = 0.10
TEST_SPLIT     = 0.10

# Background and ink
BG_COLOR      = 255      # white
INK_COLOR     = 0        # black

# Google Fonts — open license, handwriting style
# Each URL points to the raw TTF file on Google's GitHub
FONT_URLS = {
    "caveat_regular":      "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat-Regular.ttf",
    "caveat_bold":         "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat-Bold.ttf",
    "indie_flower":        "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "patrick_hand":        "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "shadows_into_light":  "https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight-Regular.ttf",
    "dancing_script":      "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript-Regular.ttf",
    "kalam_regular":       "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf",
    "architects_daughter": "https://github.com/google/fonts/raw/main/ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf",
}

# Font sizes to simulate different handwriting scales
FONT_SIZES = [32, 36, 40]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(path: Path) -> list:
    """
    Load sentences from the corpus file.

    WHAT IT DOES:
      Reads the corpus line by line, strips comment lines (starting with #)
      and blank lines, then deduplicates. Returns a clean list of sentences.

    WHY DEDUPLICATE?
      Duplicate sentences inflate training data without adding new information.
      The model sees the same sentence twice, learns it more strongly than
      others, and becomes biased toward it. Deduplication ensures balanced
      representation across all sentence types.

    WHY STRIP COMMENTS?
      The corpus uses # headers for human readability. The model should never
      see a # line — it's not a sentence, and training on it would teach the
      model that # is valid handwritten text.
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and not text.startswith("#"):
                sentences.append(text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in sentences:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"  Loaded {len(unique)} unique sentences from corpus")
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD ABBREVIATION DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════

def load_abbreviations(path: Path) -> dict:
    """
    Load the abbreviation expansion dictionary.

    WHAT IT DOES:
      Reads the JSON abbreviation file and flattens all categories
      (english_abbreviations, tagalog_abbreviations, common_taglish)
      into a single lookup dictionary.

    WHY FLATTEN?
      At expansion time, we don't care which category an abbreviation
      belongs to — we just need to look it up. Keeping categories separate
      is for human organization; flattening is for machine efficiency.

    WHY INCLUDE THIS IN PHASE 2?
      We label each generated image with both the original sentence (with
      abbreviations) AND the expanded version. In Phase 3, we can train
      TrOCR to output expanded text directly — not just transcribe what
      it sees, but understand what it means.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat = {}
    for category, entries in data.items():
        if category == "_metadata":
            continue
        if isinstance(entries, dict):
            flat.update(entries)

    print(f"  Loaded {len(flat)} abbreviation entries")
    return flat


def expand_abbreviations(text: str, abbrev_dict: dict) -> str:
    """
    Expand abbreviations in a sentence using the dictionary.

    WHAT IT DOES:
      Splits the sentence into words, looks each word up in the
      abbreviation dictionary (case-insensitive), and replaces matches.

    WHY WORD-LEVEL ONLY?
      Substring matching (e.g., replacing "pls" inside "please") would
      corrupt valid words. Word-level matching is safer and more accurate.

    LIMITATION:
      Context-dependent abbreviations (e.g., "re" could be "regarding"
      or the musical note) are always expanded to the default. A more
      sophisticated context-aware expander is planned for Phase 4.
    """
    words = text.split()
    expanded = []
    for word in words:
        key = word.lower().rstrip(".,;:!?")
        if key in abbrev_dict:
            # Preserve trailing punctuation
            punct = word[len(key):]
            expanded.append(abbrev_dict[key] + punct)
        else:
            expanded.append(word)
    return " ".join(expanded)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DOWNLOAD AND LOAD FONTS
# ══════════════════════════════════════════════════════════════════════════════

def download_fonts(fonts_dir: Path) -> dict:
    """
    Download handwriting fonts and load them at multiple sizes.

    WHAT IT DOES:
      Downloads each font TTF from Google Fonts GitHub (open license).
      Loads each font at every size in FONT_SIZES.
      Returns a dict of {variant_name: ImageFont object}.

    WHY MULTIPLE SIZES?
      Real handwriting varies in scale. Some people write large, some small.
      Training on multiple font sizes teaches the model to recognize
      characters regardless of their absolute pixel size.

    WHY IDEMPOTENT (if not font_path.exists())?
      Colab sessions reset. If you re-run this notebook after a disconnect,
      you don't want to re-download 8 fonts unnecessarily. The existence
      check makes this function safe to call multiple times.

    WHY GOOGLE FONTS SPECIFICALLY?
      All Google Fonts are open-source under the SIL Open Font License.
      Commercial use is permitted. This is critical for an industry product
      — using unlicensed fonts would be a legal liability.
    """
    fonts_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}

    print(f"  Downloading {len(FONT_URLS)} fonts...")
    for name, url in FONT_URLS.items():
        font_path = fonts_dir / f"{name}.ttf"

        if not font_path.exists():
            try:
                urllib.request.urlretrieve(url, font_path)
                print(f"    ✓ Downloaded: {name}")
            except Exception as e:
                print(f"    ✗ Failed: {name} — {e}")
                continue

        for size in FONT_SIZES:
            try:
                key = f"{name}_s{size}"
                loaded[key] = ImageFont.truetype(str(font_path), size)
            except Exception as e:
                print(f"    ✗ Could not load {name} at size {size}: {e}")

    print(f"  Loaded {len(loaded)} font variants "
          f"({len(FONT_URLS)} fonts × {len(FONT_SIZES)} sizes)")
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RENDER LINE IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_line(text: str, font: ImageFont.FreeTypeFont,
                height: int = IMG_HEIGHT,
                max_width: int = MAX_WIDTH) -> Image.Image:
    """
    Render a single text line as a grayscale image.

    WHAT IT DOES:
      1. Creates a white canvas of (max_width × height)
      2. Draws the text centered vertically
      3. If text overflows width, expands canvas horizontally
      4. Resizes back to (max_width × height)

    WHY FIXED HEIGHT, DYNAMIC WIDTH?
      TrOCR uses a fixed-height encoder that processes the full image width
      as a sequence. Fixed height = consistent feature extraction.
      Dynamic width = no text gets cut off.

    WHY RESIZE BACK TO MAX_WIDTH?
      After dynamic expansion, we resize back so all images have the same
      dimensions. This is required for batching — PyTorch DataLoader
      stacks images into tensors, which requires identical shapes.

    WHY CENTER VERTICALLY?
      Some fonts have large ascenders or descenders. Without centering,
      letters like 'g' or 'p' with descenders would be cut off at the
      bottom, and the model would learn incomplete letter shapes.
    """
    # Temporary canvas to measure text
    tmp = Image.new("L", (max_width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Expand canvas if text is wider
    canvas_w = max(max_width, text_w + 40)
    img = Image.new("L", (canvas_w, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Center text
    x = (canvas_w - text_w) // 2 - bbox[0]
    y = (height - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=INK_COLOR, font=font)

    # Resize to standard width
    img = img.resize((max_width, height), Image.LANCZOS)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_line(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Apply random augmentation to simulate real handwriting variation.

    WHAT IT DOES:
      Applies a random combination of 7 transformations, each simulating
      a real physical condition that affects how handwriting looks in
      scanned or photographed documents.

    WHY AUGMENT AT ALL?
      279 sentences × 8 font variants = ~2,200 unique images.
      That is far too few to train a robust model. Augmentation multiplies
      the effective dataset size by creating plausible variations of each
      image. The model sees the same sentence rendered differently each
      time, forcing it to learn the underlying text rather than memorizing
      specific pixel patterns.

    WHY USE rng (a seeded Random instance) INSTEAD OF random.random()?
      A seeded Random instance produces the same sequence of random numbers
      every run. This means augmentation is reproducible — re-running the
      generator produces the exact same images. This is essential for
      debugging and for team members to verify identical datasets.

    THE 7 AUGMENTATIONS EXPLAINED:

    1. ROTATION (±10°)
       Real handwriting is rarely perfectly horizontal. People tilt the
       paper or their writing hand. 10° covers the vast majority of
       real-world tilt without making text unreadable.

    2. SHEAR (±0.12)
       Shear simulates italic-style writing slant. Some people write with
       a strong forward lean, others write upright. Without shear training,
       the model fails on slanted handwriting.

    3. INK DARKNESS VARIATION (0.55–1.0)
       Pen pressure varies. A ballpoint pen on the last stroke of a word
       runs lighter than the first. We simulate this by randomly darkening
       only the ink pixels (those below brightness 200), leaving the white
       background untouched.

    4. GAUSSIAN BLUR (40% chance, radius 0.3–1.5)
       Out-of-focus phone cameras produce blurred images. Scanners at low
       DPI produce soft edges. Blur teaches the model to handle imperfect
       image quality without breaking.

    5. GAUSSIAN NOISE (50% chance, σ=3–10)
       Paper has grain. Old documents have speckle. Noise teaches the model
       to separate signal (ink strokes) from background noise.

    6. TRANSLATION (±6px horizontal, ±4px vertical)
       When a document is photographed or cropped, the text is never
       perfectly centered. Small random shifts teach position invariance.

    7. BRIGHTNESS JITTER (±20)
       Scanner exposure settings vary. A slightly overexposed scan looks
       brighter. A slightly underexposed scan looks darker. This teaches
       overall illumination robustness.
    """
    img = img.copy()
    w, h = img.size

    # 1. Rotation
    angle = rng.uniform(-10, 10)
    img = img.rotate(angle, fillcolor=BG_COLOR, expand=False)

    # 2. Shear
    shear = rng.uniform(-0.12, 0.12)
    img = img.transform(
        (w, h), Image.AFFINE,
        (1, shear, -shear * h / 2, 0, 1, 0),
        fillcolor=BG_COLOR
    )

    # 3. Ink darkness variation
    factor = rng.uniform(0.55, 1.0)
    arr = np.array(img, dtype=np.float32)
    ink = arr < 200
    arr[ink] = arr[ink] * factor
    img = Image.fromarray(arr.astype(np.uint8))

    # 4. Gaussian blur
    if rng.random() < 0.4:
        radius = rng.uniform(0.3, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 5. Gaussian noise
    if rng.random() < 0.5:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, rng.uniform(3, 10), arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # 6. Translation
    tx = rng.randint(-6, 6)
    ty = rng.randint(-4, 4)
    img = img.transform(
        (w, h), Image.AFFINE,
        (1, 0, tx, 0, 1, ty),
        fillcolor=BG_COLOR
    )

    # 7. Brightness jitter
    arr = np.array(img, dtype=np.float32)
    jitter = rng.uniform(-20, 20)
    arr = np.clip(arr + jitter, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    return img


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — BUILD DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(sentences: list, abbrev_dict: dict, fonts: dict,
                  output_dir: Path, augment_factor: int = AUGMENT_FACTOR,
                  seed: int = RANDOM_SEED) -> list:
    """
    Generate the full synthetic line image dataset.

    WHAT IT DOES:
      For each sentence × each font variant × augment_factor:
        1. Render the clean base image
        2. Apply augmentation (aug_idx=0 is always the clean base)
        3. Randomly assign to train/val/test split
        4. Save to the appropriate folder
        5. Record metadata in a list of dicts

    WHY aug_idx=0 IS ALWAYS CLEAN?
      The model needs at least one perfect example of every sentence.
      If every sample is augmented, there's a small chance all samples
      of a sentence are heavily distorted and the model never sees it
      cleanly. The clean base guarantees a quality anchor per sentence.

    WHY RANDOM SPLIT PER IMAGE (NOT PER SENTENCE)?
      If we split per sentence, the model trains on clean "Pls submit"
      and tests on clean "Pls submit" — it memorizes the exact sentence.
      Random per-image splitting means the model might train on rotated
      "Pls submit" and test on noisy "Pls submit" — a fairer evaluation.

    WHAT THE RECORDS LIST CONTAINS:
      Each record is a dict with:
        - id: unique integer
        - filename: relative path to the image
        - split: train/val/test
        - text: original sentence (with abbreviations, as written)
        - expanded_text: abbreviation-expanded version
        - font: font variant name
        - aug_idx: augmentation index (0 = clean)
        - language_type: English/Tagalog/Taglish/Abbreviation/etc.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Create split directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    records = []
    img_id  = 0
    font_list = list(fonts.items())

    total_expected = len(sentences) * len(font_list) * augment_factor
    print(f"\n  Generating images:")
    print(f"    {len(sentences)} sentences")
    print(f"    × {len(font_list)} font variants")
    print(f"    × {augment_factor} augmentations")
    print(f"    = ~{total_expected:,} total images\n")

    for sentence in tqdm(sentences, desc="  Rendering lines"):
        expanded = expand_abbreviations(sentence, abbrev_dict)

        for font_name, font in font_list:
            # Render clean base image once per sentence/font pair
            try:
                base_img = render_line(sentence, font)
            except Exception as e:
                print(f"\n  Warning: Could not render '{sentence[:30]}' "
                      f"with {font_name}: {e}")
                continue

            for aug_idx in range(augment_factor):
                # aug_idx == 0 is always the clean base
                if aug_idx == 0:
                    final_img = base_img.copy()
                else:
                    final_img = augment_line(base_img, rng)

                # Assign to split
                r = rng.random()
                if r < TEST_SPLIT:
                    split = "test"
                elif r < TEST_SPLIT + VAL_SPLIT:
                    split = "val"
                else:
                    split = "train"

                # Save image
                filename = f"{img_id:07d}.png"
                rel_path = f"images/{split}/{filename}"
                final_img.save(output_dir / rel_path)

                records.append({
                    "id":            img_id,
                    "filename":      rel_path,
                    "split":         split,
                    "text":          sentence,
                    "expanded_text": expanded,
                    "font":          font_name,
                    "aug_idx":       aug_idx,
                })
                img_id += 1

    print(f"\n  Generated {img_id:,} images total")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE LABELS CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_labels(records: list, output_dir: Path) -> dict:
    """
    Save all image records to a labels.csv file.

    WHAT IT DOES:
      Writes a CSV with one row per image containing all metadata.
      Also computes and prints split size statistics.

    WHY CSV (NOT JSON OR DATABASE)?
      CSV is readable by pandas, Excel, Google Sheets, and every ML
      framework. It's the lowest common denominator for tabular data.
      A junior developer can open it and inspect it immediately.
      JSON would be valid too but harder to scan row by row.

    WHY INCLUDE expanded_text AS A COLUMN?
      In Phase 3, we can choose whether to train TrOCR to:
        A) Output the literal text it sees (use 'text' column)
        B) Output the expanded meaning (use 'expanded_text' column)
      Having both columns in the CSV means we can switch strategies
      without re-generating the dataset. This is called future-proofing.
    """
    csv_path = output_dir / "labels.csv"
    fieldnames = ["id", "filename", "split", "text",
                  "expanded_text", "font", "aug_idx"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    splits = {"train": 0, "val": 0, "test": 0}
    for r in records:
        splits[r["split"]] += 1

    print(f"\n  Labels saved → {csv_path}")
    print(f"  Split summary:")
    print(f"    train : {splits['train']:,}")
    print(f"    val   : {splits['val']:,}")
    print(f"    test  : {splits['test']:,}")
    print(f"    total : {sum(splits.values()):,}")
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def save_manifest(sentences: list, fonts: dict, records: list,
                  splits: dict, output_dir: Path) -> dict:
    """
    Save a versioned dataset manifest JSON.

    WHAT IT DOES:
      Records every parameter that defines this dataset — seed, split
      ratios, font list, augmentation pipeline, image dimensions,
      sentence count, and total images. Saves as JSON.

    WHY THIS IS NON-NEGOTIABLE:
      Six months from now, if a model underperforms, you need to know
      exactly what data it trained on. The manifest is the dataset's
      birth certificate. Without it, reproducibility is impossible.

    WHY SEMANTIC VERSIONING (2.0.0)?
      Version 1.0.0 was the old single-word dataset (now obsolete).
      This new line-level dataset is a breaking change in format and
      purpose — it warrants a major version bump to 2.0.0.
      MAJOR.MINOR.PATCH: major = breaking change, minor = new features,
      patch = bug fixes.
    """
    manifest = {
        "version":          "2.0.0",
        "phase":            2,
        "created_at":       datetime.utcnow().isoformat() + "Z",
        "random_seed":      RANDOM_SEED,
        "task":             "taglish-handwritten-document-reading",
        "input_level":      "line",
        "languages":        ["English", "Tagalog", "Taglish"],
        "abbreviations":    True,
        "num_sentences":    len(sentences),
        "num_fonts":        len(FONT_URLS),
        "num_font_variants":len(fonts),
        "font_sizes":       FONT_SIZES,
        "augment_factor":   AUGMENT_FACTOR,
        "image_height":     IMG_HEIGHT,
        "image_max_width":  MAX_WIDTH,
        "splits":           splits,
        "total_images":     sum(splits.values()),
        "fonts_used":       list(FONT_URLS.keys()),
        "augmentation_pipeline": [
            "RandomRotation(±10°)",
            "RandomShear(±0.12)",
            "InkDarknessVariation(0.55–1.0)",
            "GaussianBlur(p=0.4, radius=0.3–1.5)",
            "GaussianNoise(p=0.5, σ=3–10)",
            "RandomTranslation(±6px, ±4px)",
            "BrightnessJitter(±20)",
        ],
        "label_columns": {
            "text":          "original sentence with abbreviations as written",
            "expanded_text": "abbreviation-expanded version for understanding tasks",
        },
        "downstream_model": "microsoft/trocr-base-handwritten",
        "notes": (
            "Line-level synthetic dataset for Taglish handwritten document reading. "
            "aug_idx=0 is always the clean base render. "
            "Abbreviation expansion uses data/sentences/abbreviations.json. "
            "Compatible with HuggingFace TrOCR fine-tuning pipeline (Phase 3)."
        )
    }

    path = output_dir / "dataset_manifest_phase2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  Manifest saved → {path}")
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(output_dir: Path, records: list, n: int = 5):
    """
    Verify n random images are valid before declaring Phase 2 complete.

    WHAT IT CHECKS:
      - File exists on disk
      - Image opens without error
      - Image dimensions match expected (MAX_WIDTH × IMG_HEIGHT)
      - Image mode is grayscale (L)

    WHY THIS MATTERS:
      In a generation loop of 10,000+ images, one rendering error might
      silently produce a corrupt file. Training would then crash hours
      later with a cryptic PIL error. The sanity check catches this at
      the source. Fail fast, fail loudly.
    """
    print("\n  Running sanity checks...")
    rng = random.Random(0)
    samples = rng.sample(records, min(n, len(records)))

    for rec in samples:
        img_path = output_dir / rec["filename"]
        assert img_path.exists(), f"Missing: {img_path}"
        img = Image.open(img_path)
        assert img.size == (MAX_WIDTH, IMG_HEIGHT), \
            f"Wrong size {img.size} for {img_path}"
        assert img.mode == "L", f"Expected grayscale, got {img.mode}"
        print(f"    ✓ '{rec['text'][:40]}' | {rec['font']} "
              f"| aug={rec['aug_idx']} | split={rec['split']}")

    print("  All sanity checks passed.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    corpus_path   = CORPUS_PATH,
    abbrev_path   = ABBREV_PATH,
    output_dir    = OUTPUT_DIR,
    fonts_dir     = FONTS_DIR,
    augment_factor= AUGMENT_FACTOR,
):
    """
    Full Phase 2 pipeline. Call this from the Colab notebook.
    Returns (records, manifest).

    WHAT IT DOES:
      Orchestrates all 8 steps in order:
        1. Load corpus
        2. Load abbreviation dictionary
        3. Download and load fonts
        4. Generate images (render + augment + split)
        5. Save labels CSV
        6. Save manifest
        7. Sanity check

    WHY A SINGLE ORCHESTRATOR FUNCTION?
      This is the Facade pattern. The notebook only needs to call one
      function. All complexity is hidden inside. A new team member can
      read the notebook and understand the pipeline without reading this
      entire file. They only read this file when they need to modify
      a specific step.
    """
    print("\n" + "═" * 60)
    print("  Phase 2 — Taglish Line Dataset Generator")
    print("  Handwritten Document Reader")
    print("═" * 60)

    corpus_path  = Path(corpus_path)
    abbrev_path  = Path(abbrev_path)
    output_dir   = Path(output_dir)
    fonts_dir    = Path(fonts_dir)

    # Step 1
    print("\n[1/7] Loading sentence corpus...")
    sentences = load_corpus(corpus_path)

    # Step 2
    print("\n[2/7] Loading abbreviation dictionary...")
    abbrev_dict = load_abbreviations(abbrev_path)

    # Step 3
    print("\n[3/7] Setting up fonts...")
    fonts = download_fonts(fonts_dir)
    if not fonts:
        raise RuntimeError(
            "No fonts loaded. Check internet connection and font URLs."
        )

    # Step 4
    print("\n[4/7] Generating synthetic line images...")
    records = build_dataset(
        sentences, abbrev_dict, fonts,
        output_dir, augment_factor
    )

    # Step 5
    print("\n[5/7] Saving labels CSV...")
    splits = save_labels(records, output_dir)

    # Step 6
    print("\n[6/7] Saving dataset manifest...")
    manifest = save_manifest(sentences, fonts, records, splits, output_dir)

    # Step 7
    print("\n[7/7] Running sanity checks...")
    sanity_check(output_dir, records)

    print("\n" + "═" * 60)
    print("  Phase 2 Complete")
    print(f"  Total images    : {manifest['total_images']:,}")
    print(f"  Sentences       : {manifest['num_sentences']}")
    print(f"  Font variants   : {manifest['num_font_variants']}")
    print(f"  Augmentations   : {manifest['augment_factor']}x")
    print(f"  Output          : {output_dir}")
    print("═" * 60 + "\n")

    return records, manifest


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    records, manifest = build_pipeline()
