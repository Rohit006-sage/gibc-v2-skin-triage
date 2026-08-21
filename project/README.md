# Skin Lesion Triage Tool (HAM10000, GIBC V2 Track 02)

A research demo binary classifier for dermoscopy images, built around the
public HAM10000 dataset. The model classifies lesions as benign or
malignant and surfaces a Grad-CAM heatmap showing which regions of the
image drove its prediction.

**This is a hackathon research prototype, not a medical device.**

---

## Progress tracker (updated Aug 21, 2026)

**🎉 CODE COMPLETE — All development phases done. Only submission logistics remain.**

### Development (all ✅)
- [x] Phase 0.1 — HAM10000 downloaded into `data/raw/` (metadata + both image parts)
- [x] Phase 0.2 — Local venv + deps installed, `python -m src.data` passes (train=6991, val=1504, test=1520)
- [x] Phase 1 — Baseline training loop & checkpoint (`runs/baseline/best.pt`)
- [x] Phase 2/3 — v2 (RandAugment, cosine, label smoothing) + v3 refinement (LR 1e-5, cosine): **test AUROC 0.911** with TTA + temp calibration (T=3.72); 90% specificity → **sensitivity 0.887**
- [x] Phase 3 — Grad-CAM engine tested, overlays generated (`outputs/gradcam/`)
- [x] Phase 4 — Test-set evaluation executed, all plots saved (`outputs/eval/`)
- [x] Phase 5 — Streamlit web app with curated sample selector (`streamlit run src/app.py`)
- [x] Phase 6 — Demo video script & guide prepared (`DEMO_VIDEO_SCRIPT.md`)
- [x] Phase 0.3 — GitHub repo public: https://github.com/Rohit006-sage/-gibc-v2-skin-triage
- [x] Bonus — Multimodal fusion (image + age/sex/site): test AUROC 0.905

### Submission (you do these ⬜)
- [ ] Phase 0.4 — Create Devpost account + start project entry (Track 02)
- [ ] Phase 7 — Record 3-min demo video (YouTube unlisted) per `DEMO_VIDEO_SCRIPT.md`
- [ ] Phase 8 — Fill 6 Devpost fields using `DEVPOST_DESCRIPTION.md` + repo URL + video + screenshots → **Submit by Sep 21, 11:45pm Taipei**

**Next action**: Create Devpost entry → record demo video → submit. All metrics, plots, and code are finalized.

---

## What you need to do (solo, 10 weeks)

This README doubles as your build checklist. Every step below is something
**you** must do — the AI agent cannot download datasets, train on a GPU,
record video, or submit on your behalf.

### Phase 0 — Repo + accounts (1 day)

1. Create a GitHub account if you don't have one.
2. Push this `project/` folder contents to your repository: `https://github.com/Rohit006-sage/-gibc-v2-skin-triage`.
   The Devpost submission requires a public repo URL.
3. Create a Devpost account at https://gibc-v2.devpost.com and start a
   project entry for **Track 02: Applied (Medical Technology & Finance)**.
4. Join the official Discord (https://discord.gg/4y6RghaSt5) for support.

### Phase 1 — Data (week 1)

5. Download **HAM10000** ("Skin Cancer MNIST: HAM10000") from one of:
   - Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Harvard Dataverse (official source)

   **Already done** — `data/raw/` contains the metadata CSV and both image
   parts. The extra `hmnist_*.csv` files in there are unused by the pipeline
   and can be deleted or left as-is (they're gitignored).
6. Unzip and place the files so the layout matches what `src/data.py`
   expects:
   ```
   project/data/raw/HAM10000_metadata.csv
   project/data/raw/HAM10000_images_part_1/   <-- 5000 .jpg files
   project/data/raw/HAM10000_images_part_2/   <-- 5015 .jpg files
   ```
7. From inside the `project/` directory:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -m src.data
   ```
   You should see three `[OK]` lines confirming the split is patient-safe
   and a sample loads correctly. **Paste the output to the AI agent** if
   anything fails.

### Phase 2 — Baseline training (week 2)

8. Open `src/train.py` and read it. Confirm: BCE loss with pos_weight,
   AdamW, val AUROC tracked, best checkpoint saved.
9. Upload `project/` to **Kaggle Notebooks** (free GPU) or open in
   **Google Colab**. Mount or copy the HAM10000 data so `data/raw/` is
   populated in the notebook environment.
10. Run:
    ```bash
    python -m src.train \
      --metadata data/raw/HAM10000_metadata.csv \
      --image-dir data/raw/HAM10000_images_part_1 \
      --image-dir data/raw/HAM10000_images_part_2 \
      --output-dir runs/baseline \
      --arch efficientnet_b0 \
      --epochs 10 \
      --batch-size 32 \
      --lr 1e-4
    ```
11. After training, check `runs/baseline/training_log.csv` and confirm
    the val AUROC curve climbed. Target: ≥ 0.88. **Paste the final
    numbers to the AI agent** and ask for the next iteration.

### Phase 3 — Iterating the model (week 3)

12. Ask the AI agent for an outline of week-3 changes (drop backbone
    freeze, try a different architecture, focal loss vs pos_weight, etc.).
    Approve the outline, then re-run training with new flags.
13. Compare `runs/v2/training_log.csv` vs `runs/baseline/`. Keep whichever
    has better val AUROC.

### Phase 4 — Interpretability (week 4)

14. Run Grad-CAM on the best checkpoint:
    ```bash
    python -m src.gradcam \
      --checkpoint runs/baseline/best.pt \
      --metadata data/raw/HAM10000_metadata.csv \
      --image-dir data/raw/HAM10000_images_part_1 \
      --image-dir data/raw/HAM10000_images_part_2 \
      --output-dir outputs/gradcam
    ```
15. Open the PNGs in `outputs/gradcam/`. Pick 3–5 compelling ones (one
    correctly-flagged malignant, one correctly-flagged benign, one or
    two failure cases). These become **Devpost screenshots** and
    **demo-video segments**.

### Phase 5 — Test-set evaluation (week 7 — do this before app polish)

16. Run the held-out test evaluation:
    ```bash
    python -m src.evaluate \
      --checkpoint runs/v3/best.pt \
      --metadata data/raw/HAM10000_metadata.csv \
      --image-dir data/raw/HAM10000_images_part_1 \
      --image-dir data/raw/HAM10000_images_part_2 \
      --output-dir runs/v3 --tta --calibrate
    ```
    The `--tta` flag averages original + horizontally-flipped logits and
    `--calibrate` fits a temperature-scaling parameter on the val set.
17. Open `runs/v3/test_metrics.json` — it now includes AUROC, sensitivity at
    95% specificity, calibration stats, and operating points (threshold tuned
    on val, metrics reported on test).
18. Save 3 screenshots for the Devpost submission from this evaluation:
    - AUROC + loss curves (from `training_log.csv`, plot with matplotlib)
    - One Grad-CAM overlay on a malignant case
    - Confusion matrix or per-class report (render as image)

### Phase 6 — Web app (weeks 5–6)

19. Run locally first:
    ```bash
    streamlit run src/app.py
    ```
    Open the URL it prints, upload a sample image, confirm you see a
    prediction + heatmap.
20. Deploy the app publicly so judges can click a link:
    - **Streamlit Cloud** (easiest): connect your GitHub repo, point at
      `src/app.py`.
    - **Hugging Face Spaces** (alt): create a Space with Streamlit SDK,
      push the same files.
21. Test the public URL on your phone. Screenshot the working app for
    Devpost deliverable 6.

### Phase 7 — Demo video (weeks 8–9)

22. Record the demo video following the structure in
    `DEVPOST_DESCRIPTION.md`:
    - 30s — problem statement
    - 45s — architecture diagram (use draw.io or Excalidraw, export PNG)
    - 90s — live demo of the deployed app
    - 45s — results table + Grad-CAM examples
    - 30s — limitations + impact
23. Upload to YouTube as **unlisted**, paste the link into Devpost.
    Private is not accepted.

### Phase 8 — Submission (week 10)

24. Fill all six Devpost fields:
    - **Project Description** — use `DEVPOST_DESCRIPTION.md` (with real
      numbers filled in)
    - **Source Code** — your GitHub repo URL
    - **Demo Video** — YouTube unlisted link
    - **Built With** — copy from the section below
    - **Team Information** — your real full name (solo entry)
    - **Screenshots** — 3+ images from week 7
25. Submit **at least 24 hours before** the Sep 21, 11:45pm Taipei
    deadline. Late submissions are not accepted.

---

## Repository layout

```
project/
  data/raw/                 HAM10000 metadata + image folders (you provide)
  models/                   Deployed checkpoint + calibration.json (committed) ✅ ready for Streamlit Cloud / HF Spaces
  src/
    data.py                 dataset class + patient-grouped split
    transforms.py           train/val augmentation pipelines
    model.py                timm-based binary classifier
    train.py                training loop (cosine LR, label smoothing, early stop)
    resume.py               fine-tune an existing checkpoint for more epochs
    multimodal.py           image + tabular (age/sex/site) fusion model
    evaluate.py             test metrics + TTA + calibration + operating points
    gradcam.py              Grad-CAM overlay generator
    app.py                  Streamlit web demo (TTA + calibrated probabilities)
  configs/                  (reserved for future hyperparam configs)
  runs/                     (created by train.py; not committed)
  outputs/                  (created by gradcam.py; not committed)
  requirements.txt
  README.md                 (this file)
  DEVPOST_DESCRIPTION.md    (template for the Devpost write-up)
  DEMO_VIDEO_SCRIPT.md      (demo video structure & talking points)
```

## Built With (paste into Devpost's "Built With" field)

- Python 3.11
- PyTorch 2.x, torchvision
- timm (EfficientNet-B0 backbone)
- torchvision transforms
- scikit-learn (StratifiedGroupKFold, AUROC, classification metrics)
- pandas, NumPy, Pillow, Matplotlib
- Streamlit (web demo interface)
- HAM10000 dataset (CC BY-NC 4.0)
- AI Development Assistant: Antigravity / Gemini Coding Agent (code scaffolding, Grad-CAM integration, and documentation)

## Datasets

- **HAM10000** ("Skin Cancer: MNIST: HAM10000"), Tschandl et al., 2018.
  10,015 dermoscopy images across 7 diagnostic categories, mapped to binary triage risk (Benign vs Malignant).

## AI Assistance Disclosure (GIBC V2 Compliance)

In compliance with the GIBC V2 hackathon rules on AI tool transparency:
- AI coding assistants (Antigravity / Gemini) were utilized for code structuring, PyTorch boilerplate, Grad-CAM hook implementations, and formatting submission markdown templates.
- All machine learning architectures, patient-stratified data splitting, evaluation logic, and clinical triage design decisions were reviewed and verified.

## License & Ethics

- **Dataset License**: HAM10000 is distributed under CC BY-NC 4.0. This prototype strictly adheres to non-commercial research use.
- **De-identified Data**: All records are anonymized; no identifiable patient health information (PHI) is used.
- **Medical Disclaimer**: This software is an experimental research prototype for the GIBC V2 Hackathon. It has not been clinically validated or cleared by regulatory authorities and **must not be used for medical diagnostic decisions or patient care**.
