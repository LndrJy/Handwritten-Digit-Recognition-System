"""
Phase 3 — Inference Module
Handwritten Document Reader
═══════════════════════════════════════════════════════════════════════════════
PURPOSE
  Loads the best trained TrOCR checkpoint and runs inference on new
  handwritten line images. This is the module that Phase 5 (document
  pipeline) and Phase 6 (API) will import and call.

  Keeping inference separate from training is a professional standard:
    - Training code has many dependencies (tqdm, schedulers, etc.)
    - Inference code should be lean and fast
    - Separating them makes the API layer simple to build in Phase 6
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class HandwritingReader:
    """
    Inference wrapper around the fine-tuned TrOCR model.

    DESIGN PRINCIPLE — WHY A CLASS INSTEAD OF A FUNCTION?
      Loading the model and processor takes ~5 seconds and ~400MB of memory.
      If inference were a plain function, it would reload the model on every
      call. A class loads once in __init__ and reuses across many calls.
      This is called the Singleton pattern and is standard for ML services.

    USAGE:
      reader = HandwritingReader("checkpoints/best_model")
      text   = reader.read("path/to/image.png")
      result = reader.read_with_confidence("path/to/image.png")
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/best_model",
                 device: str = None):
        """
        Load model and processor from checkpoint.

        WHY device=None AS DEFAULT?
          Automatically detecting the device (GPU if available, else CPU)
          makes the class work identically on Colab (GPU) and a developer's
          laptop (CPU) without changing any code. This is called
          hardware-agnostic design.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_dir}\n"
                f"Run Phase 3 training first to generate the checkpoint."
            )

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"  Loading HandwritingReader from {checkpoint_dir}...")
        self.processor = TrOCRProcessor.from_pretrained(checkpoint_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            checkpoint_dir
        ).to(self.device)
        self.model.eval()
        # model.eval() disables dropout — essential for inference
        # Without this, predictions would vary on every call for the same image

        print(f"  Ready on {self.device}")

    @torch.no_grad()
    def read(self, image_input, expand_abbreviations: bool = False) -> str:
        """
        Read handwritten text from a single line image.

        PARAMETERS:
          image_input: str/Path (file path) or PIL.Image
          expand_abbreviations: if True, apply abbreviation expansion
                                (Phase 4 feature — placeholder for now)

        RETURNS:
          Transcribed text string

        WHY @torch.no_grad()?
          Disables gradient computation during inference.
          Saves ~30% memory and speeds up inference.
          Always use for inference — gradients are only needed for training.
        """
        # Accept both file paths and PIL Images
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(
                f"Expected str, Path, or PIL.Image. Got {type(image_input)}"
            )

        # Preprocess
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate
        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

        # Decode
        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return text

    @torch.no_grad()
    def read_with_confidence(self, image_input) -> dict:
        """
        Read handwritten text and return confidence score.

        WHAT IS CONFIDENCE?
          We use the mean log probability of generated tokens as a proxy
          for confidence. Higher probability = more confident prediction.
          We normalize to [0, 1] using sigmoid for interpretability.

        WHY IS THIS IMPORTANT FOR A PRODUCTION SYSTEM?
          A system that outputs wrong answers confidently is dangerous.
          A system that says "I'm 45% confident — please verify" is safe.
          The confidence score enables the human review queue in Phase 6:
          low-confidence outputs get flagged automatically.

        RETURNS:
          {
            "text": "please submit as soon as possible",
            "confidence": 0.87,
            "review_required": False
          }
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        pixel_values = self.processor(
            images=img, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate with scores
        outputs = self.model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Decode text
        text = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        # Compute mean token confidence
        if outputs.scores:
            import torch.nn.functional as F
            token_probs = [
                F.softmax(score, dim=-1).max(dim=-1).values
                for score in outputs.scores
            ]
            confidence = float(torch.stack(token_probs).mean().cpu())
        else:
            confidence = 0.0

        return {
            "text":            text,
            "confidence":      round(confidence, 4),
            "review_required": confidence < 0.75,
        }

    @torch.no_grad()
    def read_batch(self, image_inputs: list) -> list:
        """
        Read multiple line images in a single batched forward pass.

        WHY BATCH INFERENCE?
          Processing images one at a time underutilizes the GPU.
          A T4 GPU can process 16 images simultaneously as fast as 1.
          Batching is critical for production throughput —
          a document with 20 lines should take ~1 second, not 20 seconds.

        RETURNS:
          List of text strings, one per input image
        """
        imgs = []
        for inp in image_inputs:
            if isinstance(inp, (str, Path)):
                imgs.append(Image.open(inp).convert("RGB"))
            else:
                imgs.append(inp.convert("RGB"))

        pixel_values = self.processor(
            images=imgs, return_tensors="pt"
        ).pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

        texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [t.strip() for t in texts]
