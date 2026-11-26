"""
Shared helpers for the adversarial module.

Everything here works in *pixel space*: images are float tensors in [0, 1] with
shape (B, 3, 224, 224). The ViT's own normalisation ((x - 0.5) / 0.5) happens
inside PixelSpaceViT, so gradients flow all the way back to the pixels. That is
what lets the GAN (and FGSM/PGD) build perturbations that survive being saved
as a normal image.
"""

import json
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(ROOT_DIR, "models", "custom_vit_fire")
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw", "training_images")
DEFAULT_GAN_DIR = os.path.join(ROOT_DIR, "models", "advgan")
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results", "adversarial")
IMAGE_SIZE = 224


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# Target model
# ─────────────────────────────────────────────────────────────
class PixelSpaceViT(nn.Module):
    """Frozen PyroWatch ViT that takes [0, 1] RGB tensors and returns logits."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: Optional[torch.device] = None):
        super().__init__()
        from transformers import ViTForImageClassification

        self.device = device or get_device()
        self.vit = ViTForImageClassification.from_pretrained(model_dir, attn_implementation="eager")
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        cfg_path = os.path.join(model_dir, "preprocessor_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            mean = cfg.get("image_mean", mean)
            std = cfg.get("image_std", std)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        self.id2label = {int(k): v for k, v in self.vit.config.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=(x - self.mean) / self.std).logits

    @torch.no_grad()
    def last_layer_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Final-layer attention, shape (B, heads, seq, seq)."""
        out = self.vit(pixel_values=(x - self.mean) / self.std, output_attentions=True)
        return out.attentions[-1]

    def class_id(self, name: str) -> int:
        if name not in self.label2id:
            raise ValueError(f"Unknown class '{name}'. Model classes: {list(self.label2id)}")
        return self.label2id[name]


# ─────────────────────────────────────────────────────────────
# Images <-> tensors
# ─────────────────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """PIL image -> (3, H, W) float tensor in [0, 1], resized like the ViT processor (bilinear)."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """(3, H, W) tensor in [0, 1] -> PIL image (rounded to 8-bit, like a saved PNG)."""
    arr = (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Round to 8-bit levels so reported results match an image saved as PNG."""
    return (x.clamp(0, 1) * 255.0).round() / 255.0


def perturbation_to_pil(delta: torch.Tensor, eps: float) -> Image.Image:
    """Visualise a perturbation: mid-grey = 0, amplified so ±eps spans the full range."""
    eps = max(eps, 1e-8)
    vis = (delta.detach() / (2 * eps) + 0.5).clamp(0, 1)
    return tensor_to_pil(vis)


# ─────────────────────────────────────────────────────────────
# Data: mirrors train_vit.py's split so the GAN never sees ViT test images
# ─────────────────────────────────────────────────────────────
class LabelledImages(Dataset):
    """Wraps a Hugging Face imagefolder split, returning (tensor, model_label_id)."""

    def __init__(self, hf_split, dataset_names: List[str], label2id: dict, size: int = IMAGE_SIZE):
        self.ds = hf_split
        self.size = size
        # Map dataset label index -> the model's label id via the class *name*,
        # so a different folder order can never silently swap fire/no_fire.
        missing = [n for n in dataset_names if n not in label2id]
        if missing:
            raise ValueError(f"Dataset classes {missing} not in model labels {list(label2id)}")
        self.map = {i: label2id[n] for i, n in enumerate(dataset_names)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        return pil_to_tensor(ex["image"], self.size), self.map[ex["label"]]


def load_splits(data_dir: str = DEFAULT_DATA_DIR, test_size: float = 0.2, seed: int = 42,
                val_fraction: float = 0.1):
    """
    Returns (gan_train, gan_val, vit_test, class_names) as Hugging Face splits.

    vit_test is exactly the 20% split train_vit.py evaluated on (same call, same seed).
    The ViT's training 80% is further split into GAN train / GAN val; the GAN val
    split is only used to pick the best generator epoch.
    """
    from datasets import load_dataset

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Training images not found at {data_dir}. Expected class subfolders such as "
            "'fire' and 'no_fire' (the same folder used by train_vit.py)."
        )
    ds = load_dataset("imagefolder", data_dir=data_dir)["train"]
    names = ds.features["label"].names
    split = ds.train_test_split(test_size=test_size, seed=seed)
    vit_train, vit_test = split["train"], split["test"]
    inner = vit_train.train_test_split(test_size=val_fraction, seed=seed)
    return inner["train"], inner["test"], vit_test, names


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def perturbation_stats(x: torch.Tensor, x_adv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample L-inf (in /255 units), L2, and PSNR (dB)."""
    d = (x_adv - x).flatten(1)
    linf = d.abs().max(1).values * 255.0
    l2 = d.norm(p=2, dim=1)
    mse = (d ** 2).mean(1).clamp_min(1e-12)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return linf, l2, psnr


# ─────────────────────────────────────────────────────────────
# Visual outputs
# ─────────────────────────────────────────────────────────────
def attention_overlay(attn: torch.Tensor, image: torch.Tensor, opacity: float = 0.5) -> Image.Image:
    """
    CLS-token attention heatmap blended over the image. attn: (heads, seq, seq).
    Self-contained (doesn't use matplotlib.cm.get_cmap, which matplotlib >= 3.9 removed).
    """
    import matplotlib

    cls = attn.mean(0)[0, 1:].float().cpu().numpy()
    g = int(math.sqrt(cls.shape[0]))
    grid = cls[: g * g].reshape(g, g)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    h, w = image.shape[1:]
    heat = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cmap = matplotlib.colormaps["viridis"]
    colored = (cmap(np.asarray(heat) / 255.0)[:, :, :3] * 255).astype(np.float32)
    base = np.asarray(tensor_to_pil(image), dtype=np.float32)
    return Image.fromarray(np.clip(opacity * colored + (1 - opacity) * base, 0, 255).astype(np.uint8))


def save_grid(rows: List[List[Image.Image]], captions: List[List[str]], path: str,
              col_titles: Optional[List[str]] = None, cell: int = 224) -> None:
    """Save a labelled image grid (rows x cols) as PNG."""
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    title_h = 22 if col_titles else 0
    cap_h = 30
    W, H = n_cols * cell, title_h + n_rows * (cell + cap_h)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    if col_titles:
        for c, t in enumerate(col_titles):
            draw.text((c * cell + 6, 5), t, fill="black")
    for r, row in enumerate(rows):
        y0 = title_h + r * (cell + cap_h)
        for c, img in enumerate(row):
            canvas.paste(img.resize((cell, cell)), (c * cell, y0))
            cap = captions[r][c] if c < len(captions[r]) else ""
            for i, line in enumerate(cap.split("\n")[:2]):
                draw.text((c * cell + 4, y0 + cell + 2 + i * 13), line, fill="black")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)
