"""
AdvGAN attack on the PyroWatch fire-detection ViT.

Based on: Xiao et al., "Generating Adversarial Examples with Adversarial Networks"
(IJCAI 2018, arXiv:1801.02610).

How it works
------------
  x ──► Generator G ──► δ = ε·tanh(G(x))      (bounded perturbation, ||δ||∞ ≤ ε)
  x_adv = clip(x + δ, 0, 1)
  x_adv ──► frozen PyroWatch ViT  ──► L_adv    (fool the classifier)
  x_adv ──► Discriminator D       ──► L_GAN    (look like a real image)
  ||δ||₂                          ──► L_hinge  (keep the perturbation small)

  L_G = λ_adv·L_adv + λ_gan·L_GAN + λ_hinge·L_hinge

Unlike FGSM/PGD, which need gradient access to the ViT *every time* they attack an
image, the trained generator produces an adversarial image in a single forward
pass with no access to the ViT at all. That is the threat model worth discussing:
once trained, the attacker only needs G.

Usage
-----
  # untargeted: make the ViT get any image wrong
  python -m src.adversarial.advgan

  # targeted: make real fires read as "no_fire" (the dangerous failure: a missed fire)
  python -m src.adversarial.advgan --target no_fire

  python -m src.adversarial.advgan --help   # all options
"""

import argparse
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.adversarial.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_GAN_DIR,
    DEFAULT_MODEL_DIR,
    IMAGE_SIZE,
    LabelledImages,
    PixelSpaceViT,
    get_device,
    load_splits,
    perturbation_to_pil,
    quantize,
    save_grid,
    set_seed,
    tensor_to_pil,
)

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kwargs):
        return x


# ─────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    """Encoder → residual blocks → decoder (the AdvGAN generator layout). Outputs a bounded perturbation."""

    def __init__(self, eps: float, ngf: int = 16, n_res: int = 4):
        super().__init__()
        self.eps = eps
        layers = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf * 4), nn.ReLU(True),
        ]
        layers += [ResBlock(ngf * 4) for _ in range(n_res)]
        layers += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """x in [0,1] → (x_adv, delta)."""
        delta = self.eps * self.net(x * 2 - 1)
        x_adv = (x + delta).clamp(0, 1)
        return x_adv, x_adv - x


class Discriminator(nn.Module):
    """Small PatchGAN: real images vs. perturbed images."""

    def __init__(self, ndf: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x * 2 - 1)


# ─────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────
def cw_margin_loss(logits: torch.Tensor, labels: torch.Tensor, kappa: float = 0.0,
                   target: Optional[int] = None) -> torch.Tensor:
    """
    Carlini–Wagner style margin loss (what AdvGAN uses for L_adv).
      untargeted: push logit[true] below the best other logit, by margin kappa
      targeted:   push logit[target] above every other logit; samples already of the
                  target class are ignored (we only attack e.g. real fires)
    """
    n_cls = logits.shape[1]
    if target is None:
        onehot = F.one_hot(labels, n_cls).bool()
        true_logit = logits[onehot]
        best_other = logits.masked_fill(onehot, float("-inf")).max(1).values
        return (true_logit - best_other).clamp(min=-kappa).mean()

    mask = labels != target
    if mask.sum() == 0:
        return logits.sum() * 0.0
    lg = logits[mask]
    onehot_t = torch.zeros_like(lg, dtype=torch.bool)
    onehot_t[:, target] = True
    target_logit = lg[:, target]
    best_other = lg.masked_fill(onehot_t, float("-inf")).max(1).values
    return (best_other - target_logit).clamp(min=-kappa).mean()


def fooled(preds: torch.Tensor, labels: torch.Tensor, target: Optional[int]) -> torch.Tensor:
    """Boolean per sample: did the attack achieve its goal?"""
    return preds == target if target is not None else preds != labels


# ─────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────
def save_generator(G: Generator, path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": G.state_dict(), **meta}, path)


def load_generator(path: str, device: Optional[torch.device] = None):
    """Returns (generator, metadata). metadata has eps, target (name or None), etc."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    G = Generator(eps=ckpt["eps"], ngf=ckpt.get("ngf", 16), n_res=ckpt.get("n_res", 4))
    G.load_state_dict(ckpt["state_dict"])
    G.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return G, meta


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_fooling(G, target_model, loader, device, target_id, max_batches=None):
    """Fooling rate on clean-correct samples (8-bit quantised, like a saved PNG)."""
    G.eval()
    hit = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        clean_pred = target_model(x).argmax(1)
        ok = clean_pred == y
        if target_id is not None:
            ok &= y != target_id
        if ok.sum() == 0:
            continue
        x_adv, _ = G(x[ok])
        adv_pred = target_model(quantize(x_adv)).argmax(1)
        hit += fooled(adv_pred, y[ok], target_id).sum().item()
        total += ok.sum().item()
    G.train()
    return hit / max(total, 1), total


def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    target_model = PixelSpaceViT(args.model_dir, device)
    target_id = target_model.class_id(args.target) if args.target else None
    print(f"Target model classes: {target_model.id2label}")
    print(f"Mode: {'targeted -> ' + args.target if args.target else 'untargeted'}")

    gan_train, gan_val, _, names = load_splits(args.data_dir, seed=args.seed)
    train_ds = LabelledImages(gan_train, names, target_model.label2id)
    val_ds = LabelledImages(gan_val, names, target_model.label2id)
    print(f"GAN train: {len(train_ds)} images | GAN val: {len(val_ds)} images "
          f"(ViT test split held out for evaluate_attack.py)")

    workers = args.workers
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers,
                          drop_last=True, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    eps = args.eps / 255.0
    G = Generator(eps=eps, ngf=args.ngf, n_res=args.n_res).to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = f"advgan_{args.target or 'untargeted'}_eps{int(args.eps)}"
    best_path = os.path.join(out_dir, f"{tag}.pt")
    meta = {"eps": eps, "eps_255": args.eps, "ngf": args.ngf, "n_res": args.n_res,
            "target": args.target, "image_size": IMAGE_SIZE, "model_dir": args.model_dir,
            "paper": "Xiao et al. 2018, AdvGAN (arXiv:1801.02610)"}

    history, best_rate = [], -1.0
    sample_x, sample_y = next(iter(val_dl))
    sample_x, sample_y = sample_x[:4].to(device), sample_y[:4].to(device)

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        sums = {"D": 0.0, "G_gan": 0.0, "G_adv": 0.0, "G_hinge": 0.0, "fool": 0.0}
        n = 0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_adv, delta = G(x)

            # ── Discriminator: real = clean images, fake = perturbed images (LSGAN) ──
            d_real, d_fake = D(x), D(x_adv.detach())
            loss_D = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) +
                            F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # ── Generator ──
            d_fake = D(x_adv)
            loss_gan = F.mse_loss(d_fake, torch.ones_like(d_fake))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = target_model(x_adv)
            loss_adv = cw_margin_loss(logits.float(), y, kappa=args.kappa, target=target_id)
            l2 = delta.flatten(1).norm(p=2, dim=1)
            loss_hinge = (l2 - args.l2_bound).clamp(min=0).mean()
            loss_G = args.lambda_adv * loss_adv + args.lambda_gan * loss_gan + args.lambda_hinge * loss_hinge

            opt_G.zero_grad(set_to_none=True)
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            with torch.no_grad():
                f = fooled(logits.argmax(1), y, target_id)
                if target_id is not None:
                    m = y != target_id
                    f = f[m] if m.any() else f.new_zeros(0)
                sums["fool"] += f.float().mean().item() if f.numel() else 0.0
            sums["D"] += loss_D.item(); sums["G_gan"] += loss_gan.item()
            sums["G_adv"] += loss_adv.item(); sums["G_hinge"] += loss_hinge.item()
            n += 1

        avg = {k: v / max(n, 1) for k, v in sums.items()}
        val_rate, val_n = evaluate_fooling(G, target_model, val_dl, device, target_id)
        rec = {"epoch": epoch, **avg, "val_fooling_rate": val_rate, "val_samples": val_n,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"Epoch {epoch:>3} | D {avg['D']:.3f} | G_gan {avg['G_gan']:.3f} | "
              f"G_adv {avg['G_adv']:.3f} | hinge {avg['G_hinge']:.3f} | "
              f"train fool {avg['fool']*100:5.1f}% | val fool {val_rate*100:5.1f}% (n={val_n}) | "
              f"{rec['seconds']}s")

        if val_rate > best_rate:
            best_rate = val_rate
            save_generator(G, best_path, {**meta, "epoch": epoch, "val_fooling_rate": val_rate})
            print(f"   ↳ new best generator saved to {best_path}")

        # Visual progress: clean / perturbation (amplified) / adversarial
        G.eval()
        with torch.no_grad():
            sx_adv, sd = G(sample_x)
            sx_adv = quantize(sx_adv)
            cp = target_model(sample_x).argmax(1)
            ap = target_model(sx_adv).argmax(1)
        G.train()
        rows, caps = [], []
        for i in range(sample_x.shape[0]):
            rows.append([tensor_to_pil(sample_x[i]), perturbation_to_pil(sd[i], eps), tensor_to_pil(sx_adv[i])])
            caps.append([f"pred: {target_model.id2label[cp[i].item()]}", "perturbation (amplified)",
                         f"pred: {target_model.id2label[ap[i].item()]}"])
        save_grid(rows, caps, os.path.join(out_dir, "samples", f"{tag}_epoch{epoch:03d}.png"),
                  col_titles=["clean", "delta", "adversarial"])

    with open(os.path.join(out_dir, f"{tag}_history.json"), "w") as f:
        json.dump({"meta": {k: v for k, v in meta.items()}, "history": history}, f, indent=2)
    print(f"\nDone. Best val fooling rate {best_rate*100:.1f}%. Generator: {best_path}")
    print(f"Next: python -m src.adversarial.evaluate_attack --generator {os.path.relpath(best_path)}")
    return best_path


def build_parser():
    p = argparse.ArgumentParser(description="Train an AdvGAN generator against the PyroWatch ViT.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_GAN_DIR)
    p.add_argument("--target", default=None,
                   help="Class name for a targeted attack, e.g. 'no_fire' to hide fires. Omit for untargeted.")
    p.add_argument("--eps", type=float, default=8.0, help="L-inf budget in /255 units (default 8 → 8/255).")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam LR for G and D (AdvGAN paper value).")
    p.add_argument("--lambda-adv", type=float, default=10.0)
    p.add_argument("--lambda-gan", type=float, default=1.0)
    p.add_argument("--lambda-hinge", type=float, default=1.0)
    p.add_argument("--l2-bound", type=float, default=4.0,
                   help="Hinge threshold c on ||delta||_2 (max possible at eps=8/255 is ~12.2).")
    p.add_argument("--kappa", type=float, default=0.0, help="Confidence margin in the C&W loss.")
    p.add_argument("--ngf", type=int, default=16)
    p.add_argument("--n-res", type=int, default=4)
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows).")
    p.add_argument("--no-amp", action="store_true", help="Disable fp16 mixed precision on GPU.")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
