"""Streamlit web app: upload a dermoscopy image, get a malignant/benign
prediction with confidence and a Grad-CAM heatmap overlay.

Run locally:
    streamlit run src/app.py

Deploy (Hugging Face Spaces or Streamlit Cloud):
    Just point at src/app.py and the same requirements.txt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.gradcam import GradCAM, _target_layer, overlay
from src.model import build_model, SUPPORTED_ARCHS
from src.transforms import eval_transform


@st.cache_resource
def load_model(checkpoint_path: str, arch: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", arch)
    model = build_model(arch=arch, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    target_layer = _target_layer(model, arch)
    return model, target_layer


def load_calibration(model_dir: Path) -> float:
    cal_path = model_dir / "calibration.json"
    if cal_path.exists():
        try:
            return float(json.loads(cal_path.read_text()).get("temperature", 1.0))
        except (json.JSONDecodeError, TypeError):
            pass
    return 1.0


@torch.no_grad()
def predict_with_tta(model, input_tensor: torch.Tensor, temperature: float) -> float:
    logits = model(input_tensor).squeeze(1)
    logits = (logits + model(torch.flip(input_tensor, dims=[3])).squeeze(1)) / 2.0
    return float(torch.sigmoid(logits / temperature).cpu().numpy()[0])


def main() -> None:
    st.set_page_config(page_title="Skin Lesion Triage (Demo)", layout="wide")

    st.title("Skin Lesion Triage Tool (Research Demo)")
    st.warning(
        "This is a research prototype for the GIBC V2 hackathon. "
        "It is NOT a medical device and must NOT be used for real diagnostic decisions."
    )

    with st.sidebar:
        st.header("Settings")
        default_ckpt = "models/best.pt"
        if not Path(default_ckpt).exists():
            default_ckpt = "runs/v3/best.pt"
        ckpt_path = st.text_input("Checkpoint path", default_ckpt)
        arch = st.selectbox("Architecture", SUPPORTED_ARCHS, index=0)
        available_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox("Device", available_devices, index=0)

    if not Path(ckpt_path).exists():
        st.error(f"Checkpoint not found at {ckpt_path}. Train a model first via src/train.py.")
        st.stop()

    model, target_layer = load_model(ckpt_path, arch, device)
    temperature = load_calibration(Path(ckpt_path).parent)
    transform = eval_transform()
    cam_engine = GradCAM(model, target_layer)

    tab1, tab2 = st.tabs(["🖼️ Analyze Lesion", "ℹ️ About & Clinical Context"])

    with tab1:
        st.subheader("Image Input")
        input_mode = st.radio(
            "Choose image source:",
            ["Pick a curated sample from HAM10000", "Upload an image"],
            horizontal=True,
        )

        image = None
        image_name = None

        if input_mode == "Pick a curated sample from HAM10000":
            sample_options = {
                "Melanoma (Malignant) — ISIC_0024310": "ISIC_0024310",
                "Actinic Keratosis / Intraepithelial Carcinoma (Malignant) — ISIC_0027254": "ISIC_0027254",
                "Melanocytic Nevus (Benign) — ISIC_0027419": "ISIC_0027419",
                "Benign Keratosis (Benign) — ISIC_0024312": "ISIC_0024312",
                "Dermatofibroma (Benign) — ISIC_0024330": "ISIC_0024330",
            }
            chosen_sample = st.selectbox("Select sample lesion:", list(sample_options.keys()))
            sample_id = sample_options[chosen_sample]
            image_name = sample_id

            # Look for image in candidate directories
            candidate_dirs = [
                Path("data/raw/HAM10000_images_part_1"),
                Path("data/raw/HAM10000_images_part_2"),
                Path("outputs/samples"),
            ]
            for d in candidate_dirs:
                cand_file = d / f"{sample_id}.jpg"
                if cand_file.exists():
                    image = Image.open(cand_file).convert("RGB")
                    break
            if image is None:
                st.info(f"Sample file {sample_id}.jpg not found in data directories. Please use the upload option.")
        else:
            uploaded = st.file_uploader("Upload a dermoscopy image", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    image_name = uploaded.name
                except Exception as e:
                    st.error(f"Could not read image: {e}")
                    st.stop()

        if image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, caption=f"Source: {image_name}", use_container_width=True)

            input_tensor = transform(image).unsqueeze(0).to(device)
            prob = predict_with_tta(model, input_tensor, temperature)

            cam, _ = cam_engine(input_tensor)
            blended = overlay(cam, image)

            with col2:
                st.subheader("Model Assessment")
                is_malignant = prob >= 0.5
                label = "Malignant / High Suspicion" if is_malignant else "Benign / Low Suspicion"
                confidence = prob if is_malignant else (1 - prob)

                if is_malignant:
                    st.error(f"⚠️ **Triage Result: {label}**")
                else:
                    st.success(f"✅ **Triage Result: {label}**")

                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Predicted Confidence", f"{confidence * 100:.1f}%")
                col_m2.metric("Malignancy Probability", f"{prob:.3f}")

                st.caption(
                    "Recommendation: Priority clinician review for high suspicion cases. "
                    "Threshold 0.5 is standard baseline; medical triage operates at 95% specificity."
                )

                st.subheader("Grad-CAM Explainability Heatmap")
                st.image(blended, caption="Red/yellow regions contributed most strongly to the prediction", use_container_width=True)

    with tab2:
        st.markdown(
            """
            ### Clinical Context & Explainability
            - **Intended Use**: This tool acts as an automated triage assistant, surfacing visual explanations alongside risk probabilities to assist clinical workflows and speed up dermatologist reviews.
            - **Explainability**: Using Grad-CAM (Gradient-weighted Class Activation Mapping) on the final convolutional layer, the tool reveals the anatomical/lesion patterns that influenced the model.
            - **Dataset**: Trained and evaluated on 10,015 dermoscopy images from the HAM10000 dataset using a patient-stratified split to prevent patient identity leakage.
            - **Disclaimer**: *Research demonstration prototype for GIBC V2 Hackathon (Track 02). Not FDA-cleared and not a diagnostic device.*
            """
        )

    st.divider()
    st.caption("GIBC V2 Track 02 (Applied Medical Technology) | Backbone: EfficientNet-B0 | Dataset: HAM10000 (CC BY-NC 4.0)")


if __name__ == "__main__":
    main()
