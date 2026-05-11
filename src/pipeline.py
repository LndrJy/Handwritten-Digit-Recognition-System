"""
Phase 5 — Full Document Pipeline
Handwritten Document Reader
═══════════════════════════════════════════════════════════════════════════════
PURPOSE
  Orchestrates the complete end-to-end handwritten document reading pipeline:

    1. Image preprocessing  (OpenCV)
    2. Line segmentation    (OpenCV contour detection)
    3. HTR inference        (TrOCR — Phase 3)
    4. Abbreviation expansion (Phase 4 expander)
    5. Document assembly    (structured output)

  Input:  A handwritten document image (JPG, PNG, PDF page)
  Output: DocumentResult — full text, per-line breakdown, confidence

WHY IS LINE SEGMENTATION DONE WITH OPENCV AND NOT ANOTHER ML MODEL?
  A layout analysis ML model (like LayoutLM) would be more accurate for
  complex documents. However it adds a 300MB+ dependency and 1-2 seconds
  of latency. For our MVP scope (letters, notes, memos) the document
  structure is simple enough that classical computer vision handles it
  reliably. This is the minimum viable approach — we add ML layout
  analysis in v2 if it's needed.

  The classical approach works as follows:
    1. Binarize — convert grayscale to black/white (Otsu thresholding)
    2. Dilate horizontally — merge nearby characters into word blobs
    3. Find contours — each contour is a candidate text region
    4. Filter by area — remove noise (too small) and page borders (too large)
    5. Sort by y-coordinate — read top to bottom
    6. Crop each line — extract as individual image for TrOCR

CONFIDENCE SCORING
  Each line gets an individual confidence score from TrOCR's generation
  probabilities. The document-level confidence is the mean of all line
  scores. Lines below 0.75 confidence are flagged for human review.
  This is the safety mechanism that makes the system clinically appropriate.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Phase 4 expander
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.expander import AbbreviationExpander, ExpandedResult


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Image preprocessing
BINARIZE_BLOCK_SIZE  = 11    # Adaptive threshold block size (must be odd)
BINARIZE_C           = 2     # Adaptive threshold constant
DILATE_KERNEL_W      = 40    # Horizontal dilation width — merges characters
DILATE_KERNEL_H      = 1     # Keep height at 1 — don't merge lines vertically

# Line segmentation
MIN_LINE_HEIGHT      = 10    # px — ignore contours shorter than this
MAX_LINE_HEIGHT      = 200   # px — ignore contours taller than this (borders)
MIN_LINE_WIDTH       = 50    # px — ignore very short fragments
LINE_PADDING         = 4     # px — padding around each cropped line

# Confidence
REVIEW_THRESHOLD     = 0.75  # lines below this are flagged for human review

# TrOCR generation
MAX_NEW_TOKENS       = 128
NUM_BEAMS            = 4


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LineResult:
    """
    Result for a single detected text line.

    FIELDS:
      line_index    — position in document (0-based, top to bottom)
      raw_text      — TrOCR output before abbreviation expansion
      expanded_text — text after abbreviation expansion
      confidence    — TrOCR generation confidence [0.0, 1.0]
      language      — detected language of this line
      changes       — abbreviation expansions applied
      review_required — True if confidence < REVIEW_THRESHOLD
      bbox          — (x, y, w, h) of the line in the original image
    """
    line_index:     int
    raw_text:       str
    expanded_text:  str
    confidence:     float
    language:       str
    changes:        list = field(default_factory=list)
    review_required:bool = False
    bbox:           tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "line_index":     self.line_index,
            "raw_text":       self.raw_text,
            "expanded_text":  self.expanded_text,
            "confidence":     round(self.confidence, 4),
            "language":       self.language,
            "changes":        self.changes,
            "review_required":self.review_required,
            "bbox":           list(self.bbox) if self.bbox else [],
        }


@dataclass
class DocumentResult:
    """
    Complete result for a processed document.

    FIELDS:
      full_text         — all expanded lines joined with newlines
      raw_text          — all raw TrOCR lines joined with newlines
      lines             — list of LineResult, one per detected line
      num_lines         — total lines detected
      doc_confidence    — mean confidence across all lines
      review_required   — True if any line needs human review
      language          — dominant language of the document
      processing_time_s — end-to-end wall time in seconds
      processed_at      — UTC ISO timestamp
    """
    full_text:          str
    raw_text:           str
    lines:              list
    num_lines:          int
    doc_confidence:     float
    review_required:    bool
    language:           str
    processing_time_s:  float
    processed_at:       str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {
            "full_text":         self.full_text,
            "raw_text":          self.raw_text,
            "num_lines":         self.num_lines,
            "doc_confidence":    round(self.doc_confidence, 4),
            "review_required":   self.review_required,
            "language":          self.language,
            "processing_time_s": round(self.processing_time_s, 3),
            "processed_at":      self.processed_at,
            "lines":             [l.to_dict() for l in self.lines],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_input) -> np.ndarray:
    """
    Preprocess a document image for line segmentation.

    PIPELINE:
      1. Load → grayscale
      2. Deskew  — correct tilt from non-flat scanning/photography
      3. Denoise — reduce scanner grain and paper texture
      4. Binarize — convert to clean black/white

    WHY GRAYSCALE FIRST?
      Color information is irrelevant for handwriting recognition.
      Grayscale reduces memory and computation by 3x (1 channel vs 3).
      All subsequent operations work on grayscale.

    WHY DESKEW?
      A document photographed at a slight angle has tilted text lines.
      Without deskewing, our horizontal line segmentation produces
      diagonal cuts that clip letters at the top/bottom of each line.
      Deskewing rotates the image to align text lines with the horizontal.

    WHY ADAPTIVE THRESHOLDING INSTEAD OF SIMPLE THRESHOLDING?
      Simple thresholding (e.g. pixel > 128 → white) fails on images
      with uneven illumination — shadows from page curl, bright spots
      from camera flash. Adaptive thresholding computes a local threshold
      for each pixel neighborhood, making it robust to lighting variation.

    PARAMETERS:
      image_input: str/Path (file path), PIL.Image, or numpy array
    RETURNS:
      Binary numpy array (0 = black ink, 255 = white background)
    """
    # Accept multiple input types
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_input}")
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Deskew
    gray = _deskew(gray)

    # Denoise — Non-local Means denoising
    # WHY NLM OVER GAUSSIAN BLUR?
    # Gaussian blur softens everything including ink edges, hurting OCR.
    # NLM selectively removes noise while preserving sharp ink strokes.
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=BINARIZE_BLOCK_SIZE,
        C=BINARIZE_C,
    )

    return binary


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Correct document tilt using moment-based skew detection.

    HOW IT WORKS:
      1. Threshold to find text pixels
      2. Compute image moments — the "center of mass" distribution
      3. The skew angle is derived from the second-order moments
      4. Rotate to correct by that angle

    WHY MOMENTS?
      Moments-based deskew is fast (no iterative search) and works well
      for documents with consistent text orientation. It handles tilts
      up to ±45 degrees reliably.

    LIMITATION:
      Fails on documents that are intentionally rotated 90° (e.g. landscape
      photos of portrait documents). A more robust approach would detect
      the dominant line angle using Hough transforms — planned for v2.
    """
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 100:
        return gray  # Not enough pixels to estimate skew

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angle in [-90, 0)
    # Correct to [-45, 45] range for typical document skew
    if angle < -45:
        angle = 90 + angle

    # Only correct if skew is meaningful (> 0.5°)
    if abs(angle) < 0.5:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LINE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_lines(binary: np.ndarray) -> list:
    """
    Detect and extract individual text lines from a binarized document.

    ALGORITHM:
      1. Invert the binary image (text becomes white, background black)
         WHY INVERT? OpenCV contour detection finds white regions on black.
         Our binary has black text on white background, so we invert.
      2. Dilate horizontally — merge nearby ink pixels into line-shaped blobs
         WHY DILATE? Individual characters are disconnected. Dilation
         connects them into a continuous region per line, making each line
         detectable as one contour.
      3. Find contours — each contour = one candidate text line
      4. Filter by size — remove noise and page borders
      5. Sort top-to-bottom by y-coordinate
      6. Crop each line from the ORIGINAL binary (not the dilated version)
         WHY CROP FROM ORIGINAL? Dilation distorts character shapes.
         We only use the dilated image to find WHERE the lines are,
         then crop from the clean original for TrOCR input.

    RETURNS:
      List of (cropped_line_image, bbox) tuples, sorted top to bottom.
      cropped_line_image: numpy array, white background, black ink
      bbox: (x, y, w, h) coordinates in the original image
    """
    # Invert: text → white, background → black
    inverted = cv2.bitwise_not(binary)

    # Horizontal dilation kernel — merges characters across a line
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (DILATE_KERNEL_W, DILATE_KERNEL_H)
    )
    dilated = cv2.dilate(inverted, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,      # only outer contours — don't recurse into letters
        cv2.CHAIN_APPROX_SIMPLE # compress horizontal/vertical runs
    )

    # Filter and collect bounding boxes
    line_boxes = []
    img_h, img_w = binary.shape
    max_area = img_h * img_w * 0.8  # ignore anything covering >80% of page

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if h < MIN_LINE_HEIGHT:
            continue  # too short — noise
        if h > MAX_LINE_HEIGHT:
            continue  # too tall — page border or large graphic
        if w < MIN_LINE_WIDTH:
            continue  # too narrow — punctuation fragment
        if area > max_area:
            continue  # covers too much of page — likely a border

        line_boxes.append((x, y, w, h))

    # Sort top to bottom (ascending y)
    line_boxes.sort(key=lambda b: b[1])

    # Crop each line from the original binary
    lines = []
    for (x, y, w, h) in line_boxes:
        # Add padding, clamp to image bounds
        x1 = max(0, x - LINE_PADDING)
        y1 = max(0, y - LINE_PADDING)
        x2 = min(img_w, x + w + LINE_PADDING)
        y2 = min(img_h, y + h + LINE_PADDING)

        crop = binary[y1:y2, x1:x2]

        # Convert to PIL for TrOCR
        pil_crop = Image.fromarray(crop).convert("RGB")
        lines.append((pil_crop, (x1, y1, x2 - x1, y2 - y1)))

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TrOCR INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def read_lines(line_images: list, processor: TrOCRProcessor,
               model: VisionEncoderDecoderModel,
               device: torch.device) -> list:
    """
    Run TrOCR inference on a list of line images.

    WHAT IT DOES:
      Batches the line images for efficient GPU inference.
      Returns (text, confidence) for each line.

    WHY BATCH INFERENCE?
      Processing 20 lines one at a time sends 20 separate jobs to the GPU.
      Each job has overhead — memory transfer, kernel launch, etc.
      Batching sends all 20 at once, using the GPU's parallel compute
      capacity. On a T4 GPU, batching 8 lines is ~5x faster than 8
      individual calls.

    WHY BATCH SIZE 8?
      TrOCR processes 384×384px images. At batch size 8 that's
      ~4.7M pixels per forward pass — safe for a 12GB T4 GPU.
      Larger batches risk OOM (Out of Memory) errors on smaller GPUs.
      8 is a conservative but safe default.

    CONFIDENCE COMPUTATION:
      We use the mean softmax probability of generated tokens as confidence.
      A token with probability 0.99 means the model is very sure.
      A token with probability 0.6 means it could be another character.
      Mean across all tokens gives per-line confidence.
    """
    INFERENCE_BATCH = 8
    results = []

    for i in range(0, len(line_images), INFERENCE_BATCH):
        batch_imgs = [img for img, _ in line_images[i:i + INFERENCE_BATCH]]

        with torch.no_grad():
            pixel_values = processor(
                images=batch_imgs, return_tensors="pt"
            ).pixel_values.to(device)

            outputs = model.generate(
                pixel_values,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode text
        texts = processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        # Compute per-sequence confidence from token scores
        if outputs.scores:
            import torch.nn.functional as F
            # scores is a tuple of (batch_size, vocab_size) tensors, one per step
            # Stack → (num_steps, batch_size, vocab_size)
            stacked = torch.stack(outputs.scores, dim=0)
            # Softmax over vocab → probability
            probs = F.softmax(stacked, dim=-1)
            # Max prob at each step → chosen token probability
            max_probs = probs.max(dim=-1).values  # (num_steps, batch_size)
            # Mean across steps → per-sequence confidence
            confidences = max_probs.mean(dim=0).cpu().tolist()
        else:
            confidences = [0.5] * len(texts)

        for text, conf in zip(texts, confidences):
            results.append((text.strip(), float(conf)))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DOCUMENT ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_document(line_results: list, bboxes: list,
                      expander: AbbreviationExpander) -> DocumentResult:
    """
    Assemble per-line results into a complete DocumentResult.

    WHAT IT DOES:
      1. Expands abbreviations in each line's raw text
      2. Builds LineResult objects with full metadata
      3. Computes document-level statistics
      4. Determines dominant language
      5. Returns DocumentResult

    WHY EXPAND AFTER OCR (NOT DURING TRAINING)?
      We could train TrOCR to output expanded text directly (we set this
      up in Phase 2 with the expanded_text label). In practice, keeping
      expansion as a post-processing step is more flexible:
        - We can update the abbreviation dictionary without retraining
        - We can expose both raw and expanded text in the API response
        - We can tune expansion rules without touching the model

    DOMINANT LANGUAGE DETECTION:
      We count language labels across all lines and take the most common.
      A document that's 60% taglish, 30% english, 10% tagalog is
      classified as "taglish" overall.
    """
    line_objs     = []
    language_counts = {}

    for idx, ((raw_text, confidence), bbox) in enumerate(
        zip(line_results, bboxes)
    ):
        # Expand abbreviations
        expand_result = expander.expand(raw_text)

        # Track language
        lang = expand_result.language
        language_counts[lang] = language_counts.get(lang, 0) + 1

        line_obj = LineResult(
            line_index     = idx,
            raw_text       = raw_text,
            expanded_text  = expand_result.expanded,
            confidence     = confidence,
            language       = lang,
            changes        = expand_result.changes,
            review_required= confidence < REVIEW_THRESHOLD,
            bbox           = bbox,
        )
        line_objs.append(line_obj)

    # Document-level aggregation
    if not line_objs:
        return DocumentResult(
            full_text="", raw_text="", lines=[],
            num_lines=0, doc_confidence=0.0,
            review_required=False, language="unknown",
            processing_time_s=0.0,
        )

    full_text     = "\n".join(l.expanded_text for l in line_objs)
    raw_text      = "\n".join(l.raw_text for l in line_objs)
    doc_confidence= sum(l.confidence for l in line_objs) / len(line_objs)
    review_req    = any(l.review_required for l in line_objs)
    dominant_lang = max(language_counts, key=language_counts.get)

    return DocumentResult(
        full_text        = full_text,
        raw_text         = raw_text,
        lines            = line_objs,
        num_lines        = len(line_objs),
        doc_confidence   = doc_confidence,
        review_required  = review_req,
        language         = dominant_lang,
        processing_time_s= 0.0,  # set by caller
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DocumentPipeline:
    """
    End-to-end handwritten document reading pipeline.

    Loads TrOCR and the expander once, then processes any number of
    documents without reloading. This is the main class that Phase 6
    (FastAPI) will import and wrap in a REST endpoint.

    USAGE:
      pipeline = DocumentPipeline(
          checkpoint_dir = "checkpoints/best_model",
          abbrev_path    = "data/sentences/abbreviations.json",
      )
      result = pipeline.read("path/to/document.jpg")
      print(result.full_text)
      print(result.to_json())

    DESIGN:
      All four pipeline steps (preprocess, segment, read, assemble)
      are implemented as module-level functions. DocumentPipeline
      holds state (model, processor, expander) and calls them in order.
      This separation makes each step independently testable — you can
      test preprocess_image without loading the model.
    """

    def __init__(self,
                 checkpoint_dir: str = "checkpoints/best_model",
                 abbrev_path:    str = "data/sentences/abbreviations.json",
                 device:         str = None):

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_dir}\n"
                f"Run Phase 3 training first."
            )

        self.device = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"\n  Loading DocumentPipeline on {self.device}...")

        # Load TrOCR
        print("  Loading TrOCR processor...")
        self.processor = TrOCRProcessor.from_pretrained(str(checkpoint_dir))
        print("  Loading TrOCR model...")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(checkpoint_dir)
        ).to(self.device)
        self.model.eval()

        # Load expander
        abbrev_path = Path(abbrev_path)
        self.expander = AbbreviationExpander(
            abbrev_path=abbrev_path if abbrev_path.exists() else None
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model: {total_params:,} parameters")
        print(f"  Pipeline ready.\n")

    def read(self, image_input) -> DocumentResult:
        """
        Read a full handwritten document image end-to-end.

        PARAMETERS:
          image_input: file path (str/Path), PIL.Image, or numpy array

        RETURNS:
          DocumentResult with full text and per-line metadata

        TIMING:
          Processing time is measured wall-clock end-to-end, including
          preprocessing, segmentation, inference, and expansion.
          This is the number that appears in the API response and the
          Streamlit demo — it's what the client sees.
        """
        t_start = time.time()

        # Step 1 — Preprocess
        binary = preprocess_image(image_input)

        # Step 2 — Segment lines
        line_images = segment_lines(binary)

        if not line_images:
            return DocumentResult(
                full_text="", raw_text="", lines=[],
                num_lines=0, doc_confidence=0.0,
                review_required=False, language="unknown",
                processing_time_s=time.time() - t_start,
            )

        # Step 3 — Read each line with TrOCR
        line_texts = read_lines(
            line_images, self.processor, self.model, self.device
        )

        # Separate images and bboxes
        bboxes = [bbox for _, bbox in line_images]

        # Step 4 — Assemble document
        result = assemble_document(line_texts, bboxes, self.expander)
        result.processing_time_s = time.time() - t_start

        return result

    def read_bytes(self, image_bytes: bytes,
                   format: str = "JPEG") -> DocumentResult:
        """
        Read from raw image bytes. Used by the FastAPI endpoint in Phase 6.

        WHY ACCEPT BYTES?
          When a client uploads a file via HTTP POST, FastAPI receives it
          as bytes — not a file path. This method converts bytes to a PIL
          Image and passes it to read(). This keeps Phase 6 simple — the
          API just calls read_bytes() without any file I/O.
        """
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.read(img)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    checkpoint_dir: str = "checkpoints/best_model",
    abbrev_path:    str = "data/sentences/abbreviations.json",
    device:         str = None,
) -> DocumentPipeline:
    """
    Initialize and return a ready-to-use DocumentPipeline.

    Consistent with all previous phases — one import, one function call.
    """
    print("\n" + "═" * 60)
    print("  Phase 5 — Full Document Pipeline")
    print("  Handwritten Document Reader")
    print("═" * 60)

    pipeline = DocumentPipeline(
        checkpoint_dir = checkpoint_dir,
        abbrev_path    = abbrev_path,
        device         = device,
    )

    print("═" * 60)
    print("  Phase 5 ready.")
    print("═" * 60 + "\n")

    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path>")
        print("Example: python pipeline.py test_document.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    pipeline   = build_pipeline()
    result     = pipeline.read(image_path)

    print(result.to_json())
