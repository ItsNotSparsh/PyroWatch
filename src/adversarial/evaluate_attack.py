"""
Evaluate AdvGAN vs. FGSM vs. PGD on the ViT's held-out test split.

Usage
-----
  python -m src.adversarial.evaluate_attack --generator models/advgan/advgan_untargeted_eps8.pt
  python -m src.adversarial.evaluate_attack            # baselines only (no generator yet)

Writes to results/adversarial/:
  results.json     all numbers
  results.md       a table you can paste into the report
  examples.png     clean | AdvGAN perturbation | AdvGAN image | PGD image, with predictions
  attention.png    ViT attention on clean vs. attacked images

All adversarial images are rounded to 8-bit before scoring, so the numbers match what
you would get after saving the image as PNG. (JPEG compression is NOT accounted for
and will weaken every attack. That is a fair point to raise in the report.)
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

from src.adversarial.advgan import load_generator
from src.adversarial.baselines import fgsm, pgd
from src.adversarial.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_RESULTS_DIR,
    LabelledImages,
    PixelSpaceViT,
    attention_overlay,
    get_device,
    load_splits,
    perturbation_stats,
    perturbation_to_pil,
    quantize,
    save_grid,
    set_seed,
    tensor_to_pil,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarise(rec, target_id, fire_id):
    y = torch.cat(rec["y"]); cp = torch.cat(rec["clean"]); ap = torch.cat(rec["adv"])
    linf = torch.cat(rec["linf"]); l2 = torch.cat(rec["l2"]); psnr = torch.cat(rec["psnr"])
    n = len(y)
    correct = cp == y
    if target_id is None:
        eligible = correct
        success = ap != y
    else:
        eligible = correct & (y != target_id)
        success = ap == target_id
    out = {
        "samples": n,
        "clean_accuracy": correct.float().mean().item(),
        "adversarial_accuracy": (ap == y).float().mean().item(),
        "attack_success_rate": (success & eligible).sum().item() / max(eligible.sum().item(), 1),
        "eligible_samples": int(eligible.sum().item()),
        "mean_linf_255": linf.mean().item(),
        "mean_l2": l2.mean().item(),
        "mean_psnr_db": psnr[torch.isfinite(psnr)].mean().item() if torch.isfinite(psnr).any() else float("inf"),
        "ms_per_image": 1000.0 * rec["seconds"] / max(n, 1),
    }
    if fire_id is not None:
        fire = y == fire_id
        if fire.any():
            out["fire_recall_clean"] = (cp[fire] == fire_id).float().mean().item()
            out["fire_recall_adversarial"] = (ap[fire] == fire_id).float().mean().item()
    return out


def markdown_table(results, eps_255, mode):
    lines = [
        f"**Attack budget:** L∞ ≤ {eps_255:g}/255 · **Mode:** {mode} · "
        f"**Test images:** {next(iter(results.values()))['samples']}",
        "",
        "| Attack | Needs ViT gradients at attack time | Clean acc. | Adversarial acc. | Attack success | "
        "Fire recall (clean → attacked) | Mean L∞ (/255) | Mean L2 | PSNR (dB) | ms / image |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    needs = {"FGSM": "Yes (1 step)", "PGD": "Yes (every step)", "AdvGAN": "No (one generator pass)"}
    for name, r in results.items():
        fr = (f"{r['fire_recall_clean']*100:.1f}% → {r['fire_recall_adversarial']*100:.1f}%"
              if "fire_recall_clean" in r else "n/a")
        lines.append(
            f"| {name} | {needs.get(name, '')} | {r['clean_accuracy']*100:.1f}% | "
            f"{r['adversarial_accuracy']*100:.1f}% | {r['attack_success_rate']*100:.1f}% | {fr} | "
            f"{r['mean_linf_255']:.2f} | {r['mean_l2']:.2f} | {r['mean_psnr_db']:.1f} | {r['ms_per_image']:.1f} |"
        )
    lines += [
        "",
        "Attack success = share of correctly classified test images the attack flipped "
        "(targeted mode: share of correctly classified non-target images pushed into the target class).",
    ]
    return "\n".join(lines)


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    model = PixelSpaceViT(args.model_dir, device)

    G, meta = (None, {})
    if args.generator:
        G, meta = load_generator(args.generator, device)
        print(f"Loaded generator: {args.generator} (eps={meta['eps']*255:.0f}/255, target={meta.get('target')})")

    eps = (args.eps / 255.0) if args.eps is not None else meta.get("eps", 8 / 255)
    target_name = args.target if args.target is not None else meta.get("target")
    if G is not None and target_name != meta.get("target"):
        print(f"Warning: generator was trained with target={meta.get('target')} but evaluating with "
              f"target={target_name}. Its numbers won't mean what you think.")
    target_id = model.class_id(target_name) if target_name else None
    fire_id = model.label2id.get("fire")
    mode = f"targeted → {target_name}" if target_name else "untargeted"

    _, _, vit_test, names = load_splits(args.data_dir, seed=args.seed)
    test_ds = LabelledImages(vit_test, names, model.label2id)
    if args.max_samples:
        test_ds = Subset(test_ds, range(min(args.max_samples, len(test_ds))))
    dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"Evaluating on {len(test_ds)} held-out test images | eps = {eps*255:.1f}/255 | {mode}")

    attacks = {
        "FGSM": lambda x, y: fgsm(model, x, y, eps, target=target_id),
        "PGD": lambda x, y: pgd(model, x, y, eps, steps=args.pgd_steps, target=target_id),
    }
    if G is not None:
        def advgan(x, y):
            with torch.no_grad():
                return G(x)[0]
        attacks["AdvGAN"] = advgan

    recs = {k: defaultdict(list, seconds=0.0) for k in attacks}
    examples = []  # (x, y, {attack: x_adv})

    for bi, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_pred = model(x).argmax(1)
        batch_adv = {}
        for name, fn in attacks.items():
            _sync(device); t0 = time.time()
            x_adv = quantize(fn(x, y))
            _sync(device)
            recs[name]["seconds"] += time.time() - t0
            with torch.no_grad():
                adv_pred = model(x_adv).argmax(1)
            linf, l2, psnr = perturbation_stats(x, x_adv)
            for k, v in (("y", y), ("clean", clean_pred), ("adv", adv_pred),
                         ("linf", linf), ("l2", l2), ("psnr", psnr)):
                recs[name][k].append(v.detach().cpu())
            batch_adv[name] = x_adv
        # keep example images: prefer correctly classified fires (the case that matters)
        for i in range(x.shape[0]):
            if len(examples) >= args.num_examples:
                break
            if clean_pred[i] == y[i] and (fire_id is None or y[i] == fire_id or bi > 3):
                examples.append((x[i].cpu(), y[i].item(), {k: v[i].cpu() for k, v in batch_adv.items()}))
        print(f"  batch {bi + 1}/{len(dl)}", end="\r")
    print()

    results = {k: summarise(v, target_id, fire_id) for k, v in recs.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    payload = {"eps_255": eps * 255, "mode": mode, "generator": args.generator,
               "pgd_steps": args.pgd_steps, "results": results}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    table = markdown_table(results, eps * 255, mode)
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table + "\n")

    # ── Example + attention figures ──
    if examples:
        show = [a for a in ("AdvGAN", "PGD", "FGSM") if a in attacks][:2]
        rows, caps, att_rows, att_caps = [], [], [], []
        with torch.no_grad():
            for x0, yl, advs in examples:
                xb = x0.unsqueeze(0).to(device)
                p0 = model(xb).softmax(1)[0]
                c0 = p0.argmax().item()
                row = [tensor_to_pil(x0)]
                cap = [f"true: {model.id2label[yl]}\npred: {model.id2label[c0]} {p0[c0]*100:.0f}%"]
                arow = [attention_overlay(model.last_layer_attention(xb)[0], x0)]
                acap = ["clean"]
                main_name = show[0]
                row.append(perturbation_to_pil(advs[main_name] - x0, eps))
                cap.append(f"{main_name} delta (amplified)")
                for name in show:
                    xa = advs[name].unsqueeze(0).to(device)
                    pa = model(xa).softmax(1)[0]
                    ca = pa.argmax().item()
                    row.append(tensor_to_pil(advs[name]))
                    cap.append(f"{name}\npred: {model.id2label[ca]} {pa[ca]*100:.0f}%")
                    arow.append(attention_overlay(model.last_layer_attention(xa)[0], advs[name]))
                    acap.append(name)
                rows.append(row); caps.append(cap); att_rows.append(arow); att_caps.append(acap)
        save_grid(rows, caps, os.path.join(args.out_dir, "examples.png"),
                  col_titles=["clean", f"{show[0]} perturbation"] + show)
        save_grid(att_rows, att_caps, os.path.join(args.out_dir, "attention.png"),
                  col_titles=["attention: clean"] + [f"attention: {s}" for s in show])
    print(f"Saved results to {args.out_dir}")
    return payload


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate adversarial attacks on the PyroWatch ViT.")
    p.add_argument("--generator", default=None, help="Path to a trained AdvGAN .pt file.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--eps", type=float, default=None,
                   help="Budget in /255 for FGSM/PGD. Defaults to the generator's eps (or 8).")
    p.add_argument("--target", default=None,
                   help="Override target class. Defaults to whatever the generator was trained for.")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None, help="Evaluate on only the first N test images.")
    p.add_argument("--num-examples", type=int, default=6)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
