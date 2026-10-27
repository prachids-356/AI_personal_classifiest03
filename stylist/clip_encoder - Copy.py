"""CLIP wrapper (Hugging Face `transformers`, no git install needed)."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = "openai/clip-vit-base-patch32"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def open_rgb(path) -> Image.Image:
    """Open an image, honour EXIF rotation (phone photos), return RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class EncoderBase:
    """Anything with encode_images / encode_texts can be plugged in (see tests)."""

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_paths(self, paths: Iterable, batch_size: int = 32,
                     progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        paths = list(paths)
        out = []
        for i in range(0, len(paths), batch_size):
            batch = [open_rgb(p) for p in paths[i:i + batch_size]]
            out.append(self.encode_images(batch))
            if progress:
                progress(min(i + batch_size, len(paths)), len(paths))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


class ClipEncoder(EncoderBase):
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_images(self, images):
        feats = []
        for i in range(0, len(images), 32):
            batch = [im.convert("RGB") for im in images[i:i + 32]]
            px = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device)
            with self.torch.no_grad():
                vis = self.model.vision_model(pixel_values=px)
                f = self.model.visual_projection(vis.pooler_output)
            feats.append(f.cpu().numpy())
        return _l2(np.concatenate(feats).astype(np.float32))

    def encode_texts(self, texts):
        inp = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with self.torch.no_grad():
            out = self.model.text_model(input_ids=inp["input_ids"],
                                        attention_mask=inp["attention_mask"])
            f = self.model.text_projection(out.pooler_output)
        return _l2(f.cpu().numpy().astype(np.float32))
