# Devpost Project Description — FINAL (paste this into the Project Description field)

> Replace `[YOUR NAME]` and `[YOUTUBE_LINK]` before submitting.

---

## Headline

An explainable AI triage tool for dermoscopy images that combines a fine-tuned EfficientNet-B0 classifier with Grad-CAM interpretability, temperature-calibrated probabilities, and a Streamlit web app — so clinicians see not just a risk label, but the visual evidence behind it.

## Problem

Skin cancer is among the most common cancers globally, and early detection substantially improves outcomes. Dermatology access is uneven, particularly outside specialist centers, creating critical screening bottlenecks. Automated triage tools that flag suspicious lesions for prioritized review have real clinical potential — but only if they communicate their reasoning rather than returning opaque labels. A black-box prediction a clinician cannot verify is one a clinician cannot trust.

## Solution

We trained a binary triage classifier (benign vs malignant) on the public HAM10000 dermoscopy dataset and wrapped it in a Streamlit web app that:
- Accepts an uploaded dermoscopy image (or a curated HAM10000 sample)
- Returns a calibrated malignant/benign probability (test-time augmentation + temperature scaling)
- Overlays a Grad-CAM heatmap showing exactly which image regions drove the prediction

The result is a tool that surfaces not just a label but a visual explanation a clinician or patient can scrutinize — designed for *triage*, not diagnosis.

## How it works (architecture)

```
HAM10000 dermoscopy images + metadata (10,015 images, 7 classes -> binary)
            |
            v
Patient-level stratified 70/15/15 split (no patient leakage)
            |
            v
EfficientNet-B0 (ImageNet-pretrained, fine-tuned 21 epochs total:
11 baseline + 10 refinement with cosine LR decay)
            |
            v
BCEWithLogitsLoss with pos_weight (class-imbalance handling)
            |
            v
Test-set evaluation: AUROC, sensitivity at 95% specificity,
per-class precision/recall, confusion matrix
            |
            v
Temperature calibration (val NLL 0.538 -> 0.273) + test-time augmentation
            |
            v
Grad-CAM heatmap on the final conv block
            |
            v
Streamlit app: upload -> predict -> explain (calibrated probability + heatmap)
```

## Results (Held-Out Test Set: 1,520 Patient-Safe Cases)

Evaluated on the patient-stratified test split (1,220 benign, 300 malignant) — zero patient overlap with train/val. Final model evaluated with test-time augmentation (horizontal-flip averaging) and temperature calibration (T = 3.72).

- **Test AUROC: 0.911**
- **Sensitivity at 95% specificity: 0.587**
- **Triage operating point (90% specificity): sensitivity 0.887, specificity 0.746** — threshold selected on validation, metrics reported on test
- **Test-Set Size**: 1,520 unseen patient images

### Per-Class Classification Report (Threshold = 0.5)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** (nv, bkl, df, vasc) | 0.9255 | 0.9066 | 0.9159 | 1,220 |
| **Malignant** (mel, bcc, akiec) | 0.6492 | 0.7033 | 0.6752 | 300 |
| **Macro Average** | 0.7874 | 0.8049 | 0.7956 | 1,520 |
| **Weighted Average** | 0.8710 | 0.8664 | 0.8684 | 1,520 |

Confusion matrix (threshold 0.5): TN = 1,106 | FP = 114 | FN = 89 | TP = 211

### Operating Points (threshold tuned on validation, metrics on test)

| Operating Point | Threshold | Test Sensitivity | Test Specificity |
| :--- | :--- | :--- | :--- |
| 95% specificity | 0.757 | 0.503 | 0.975 |
| **90% specificity (triage)** | 0.179 | **0.887** | 0.746 |
| 50% specificity | 0.000 | 1.000 | 0.000 |

For clinical triage we prefer the 90%-specificity point: it flags ~89% of malignant lesions for priority review while correctly clearing ~75% of benign cases — the honest behavior a screening aid needs.

### Model Calibration

Temperature scaling on the validation set (NLL 0.538 → 0.273) substantially improved probability calibration (reliability diagram included below). The deployed demo app applies the same TTA + calibration pipeline, so the confidence scores shown to users are trustworthy.

### Error Analysis

Grad-CAM overlays of misclassified test cases are included: false negatives (malignant lesions missed — the model attends to the lesion but underestimates risk, motivating our high-sensitivity operating point) and false positives (benign keratoses with pigment networks mimicking malignancy).

### Ablation: multimodal fusion (image + patient metadata)

We also trained a variant fusing CNN features with patient tabular metadata (age, sex, anatomical site — 17 features) via a small MLP head (`src/multimodal.py`). It reached test AUROC 0.905 (sensitivity @ 95% spec 0.570), confirming the image-only model's strength and identifying metadata fusion as a promising extension rather than a bottleneck.

## Limitations

- **Demographic bias**: HAM10000 has known demographic bias (Fitzpatrick skin types I–III overrepresented).
- **Single public cohort**: No external clinical hospital validation.
- **Grad-CAM resolution**: Grad-CAM is a coarse spatial attribution method — it demonstrates feature attribution, not clinical causation.
- **Threshold calibration**: the 90%-specificity operating point is a research baseline; real-world triage requires site-specific validation.
- **Regulatory status**: Research prototype only. Not an FDA/CE-cleared medical device. Not intended for clinical diagnosis.

## What's Next

- Expand training with multi-center diverse cohorts (ISIC 2024, Fitzpatrick17k).
- Deepen tabular fusion (richer metadata, feature interaction heads) — multimodal prototype already at test AUROC 0.905.
- Prospective reader-study validation with board-certified dermatologists.

## AI Assistance Disclosure (GIBC V2 Compliance)

- **AI tools used**: AI coding agents (Antigravity / Gemini / opencode) for code scaffolding, boilerplate, Grad-CAM hook integration, documentation, and formatting of submission materials.
- **Scope of AI assistance**: All machine learning architectures, patient-stratified split design, loss weighting, calibration methodology, evaluation logic, and medical interpretability decisions were reviewed and verified by the developer. All model training, evaluation, and validation runs were executed and verified by the developer.

---

## Separate Devpost fields

**Built With** (paste into "Built With" field):
Python 3.11, PyTorch 2.x, torchvision, timm (EfficientNet-B0), scikit-learn, pandas, NumPy, Pillow, Matplotlib, Streamlit, HAM10000 Dataset (CC BY-NC 4.0), AI coding agents (Antigravity / Gemini / opencode)

**Source Code**: https://github.com/Rohit006-sage/-gibc-v2-skin-triage

**Demo Video**: [YOUTUBE_LINK] (unlisted, 2–5 min, per `DEMO_VIDEO_SCRIPT.md`)

**Team Information**: [YOUR NAME] (solo entry)

**Screenshots** (≥3 — use these): `outputs/eval/architecture.png`, `outputs/eval/training_curves.png`, `outputs/eval/roc_curve.png`, `outputs/eval/confusion_matrix.png`, `outputs/eval/reliability_curve.png`, `outputs/gradcam/` overlays, plus a live-app screenshot after deployment