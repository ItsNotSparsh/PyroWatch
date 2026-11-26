"""
PyroWatch — Adversarial Lab (Streamlit page).

Streamlit picks this up automatically from app/pages/ when you run:
    python -m streamlit run app\\app.py
It appears in the sidebar navigation as "Adversarial Lab".
"""

import glob
import io
import os
import sys
import time

import streamlit as st
import torch
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.adversarial.advgan import load_generator
from src.adversarial.baselines import fgsm, pgd
from src.adversarial.common import (
    DEFAULT_GAN_DIR,
    DEFAULT_MODEL_DIR,
    PixelSpaceViT,
    attention_overlay,
    get_device,
    perturbation_stats,
    perturbation_to_pil,
    pil_to_tensor,
    quantize,
    tensor_to_pil,
)

st.set_page_config(page_title="PyroWatch — Adversarial Lab", page_icon="🧪", layout="wide")

st.markdown("## 🧪 Adversarial Lab")
st.caption(
    "Stress-test the fire-detection Vision Transformer. A GAN-trained generator (AdvGAN) "
    "adds a small, nearly invisible perturbation to the image so the classifier gets it wrong. "
    "FGSM and PGD are gradient-based baselines for comparison."
)


@st.cache_resource
def get_target():
    return PixelSpaceViT(DEFAULT_MODEL_DIR, get_device())


@st.cache_resource
def get_generator(path):
    return load_generator(path, get_device())


model = get_target()
device = model.device
gen_paths = sorted(glob.glob(os.path.join(DEFAULT_GAN_DIR, "*.pt")))

with st.sidebar:
    st.markdown("### ⚙️ Attack settings")
    options = (["AdvGAN"] if gen_paths else []) + ["PGD", "FGSM"]
    attack = st.radio("Attack", options, help="AdvGAN needs a trained generator in models/advgan/.")
    gen_path, meta = None, {}
    if attack == "AdvGAN":
        gen_path = st.selectbox("Generator", gen_paths, format_func=os.path.basename)
        G, meta = get_generator(gen_path)
        eps_255 = meta["eps"] * 255
        st.info(f"Budget fixed at training time: L∞ ≤ {eps_255:.0f}/255\n\n"
                f"Mode: {'targeted → ' + meta['target'] if meta.get('target') else 'untargeted'}")
        target_name = meta.get("target")
    else:
        eps_255 = st.slider("Budget ε (L∞, /255)", 1, 16, 8)
        targeted = st.checkbox("Targeted: make fires read as 'no_fire'", value=False)
        target_name = "no_fire" if targeted else None
        steps = st.slider("PGD steps", 1, 40, 10) if attack == "PGD" else 1
    if not gen_paths:
        st.warning("No AdvGAN generator found. Train one with:\n\n"
                   "`python -m src.adversarial.advgan`")

uploaded = st.file_uploader("Upload an image to attack", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a fire or forest image to begin.")
    st.stop()

img = Image.open(uploaded)
x = pil_to_tensor(img).unsqueeze(0).to(device)
target_id = model.class_id(target_name) if target_name else None
eps = eps_255 / 255.0

with torch.no_grad():
    p_clean = model(x).softmax(1)[0]
y = p_clean.argmax().unsqueeze(0)  # attack the model's own prediction (no ground truth for uploads)

if st.button("⚡ Run attack", use_container_width=True):
    t0 = time.time()
    if attack == "AdvGAN":
        with torch.no_grad():
            x_adv = G(x)[0]
    elif attack == "PGD":
        x_adv = pgd(model, x, y, eps, steps=steps, target=target_id)
    else:
        x_adv = fgsm(model, x, y, eps, target=target_id)
    x_adv = quantize(x_adv)
    elapsed = (time.time() - t0) * 1000
    with torch.no_grad():
        p_adv = model(x_adv).softmax(1)[0]
    st.session_state["adv_result"] = (uploaded.name, attack, x_adv.cpu(), p_adv.cpu(), elapsed, eps)

res = st.session_state.get("adv_result")
if not res or res[0] != uploaded.name:
    c1, c2 = st.columns(2)
    c1.image(tensor_to_pil(x[0]), caption="Input (224×224, as the ViT sees it)", use_container_width=True)
    c2.metric("Clean prediction", model.id2label[y.item()], f"{p_clean.max().item()*100:.1f}% confidence")
    st.stop()

_, used_attack, x_adv, p_adv, elapsed, used_eps = res
x_adv = x_adv.to(device)
c_clean, c_adv = p_clean.argmax().item(), p_adv.argmax().item()
flipped = c_clean != c_adv

if flipped:
    st.error(f"🚨 Attack succeeded: **{model.id2label[c_clean]}** → **{model.id2label[c_adv]}**")
else:
    st.success(f"🛡️ Model held: still **{model.id2label[c_adv]}**")

col1, col2, col3 = st.columns(3)
col1.image(tensor_to_pil(x[0]), caption=f"Clean · {model.id2label[c_clean]} {p_clean[c_clean]*100:.1f}%",
           use_container_width=True)
col2.image(perturbation_to_pil((x_adv - x)[0], used_eps),
           caption="Perturbation (amplified so it's visible)", use_container_width=True)
col3.image(tensor_to_pil(x_adv[0]), caption=f"{used_attack} · {model.id2label[c_adv]} {p_adv[c_adv]*100:.1f}%",
           use_container_width=True)

linf, l2, psnr = perturbation_stats(x, x_adv)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Max pixel change (L∞)", f"{linf.item():.1f}/255")
m2.metric("L2 norm", f"{l2.item():.2f}")
m3.metric("PSNR", f"{psnr.item():.1f} dB", help="Above ~35 dB is generally hard to see.")
m4.metric("Attack time", f"{elapsed:.0f} ms")

with st.expander("🔍 Where is the ViT looking? (attention before vs. after)"):
    a1, a2 = st.columns(2)
    a1.image(attention_overlay(model.last_layer_attention(x)[0], x[0]), caption="Clean", use_container_width=True)
    a2.image(attention_overlay(model.last_layer_attention(x_adv)[0], x_adv[0]),
             caption=used_attack, use_container_width=True)

buf = io.BytesIO()
tensor_to_pil(x_adv[0]).save(buf, format="PNG")
st.download_button("📥 Download adversarial image (PNG)", buf.getvalue(),
                   file_name=f"adversarial_{used_attack.lower()}_{os.path.splitext(uploaded.name)[0]}.png",
                   mime="image/png", use_container_width=True)
st.caption("Saved as PNG on purpose: JPEG compression partly destroys the perturbation. "
           "Re-upload the PNG in the AI Vision Lab tab to confirm the attack transfers.")

with st.expander("ℹ️ How AdvGAN works"):
    st.markdown(
        "- A **generator** network learns to output a perturbation δ with |δ| ≤ ε for any input image.\n"
        "- A **discriminator** tries to tell clean images from perturbed ones, which pushes δ to look natural.\n"
        "- The frozen PyroWatch ViT gives the **adversarial loss**: the generator is rewarded when the "
        "ViT misclassifies.\n"
        "- After training, attacking a new image is a single generator pass, with no ViT gradients needed. "
        "PGD needs many gradient steps through the ViT for every image.\n\n"
        "Reference: Xiao et al., *Generating Adversarial Examples with Adversarial Networks*, IJCAI 2018."
    )
