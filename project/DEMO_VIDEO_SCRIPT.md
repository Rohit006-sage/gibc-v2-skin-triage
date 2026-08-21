# Video Demo & Presentation Script (GIBC V2 Hackathon — Track 02)

**Target Duration**: 3 minutes (180 seconds)  
**Track**: Track 02 — Applied (Medical Technology & Finance)  
**Project**: Skin Lesion Triage Tool (HAM10000)

---

## Video Outline & Time Allocation

```
0:00 - 0:30  |  Hook & The Problem
0:30 - 1:15  |  Architecture & Technical Approach
1:15 - 2:25  |  Live Web App Demonstration (Streamlit + Grad-CAM)
2:25 - 2:45  |  Results & Clinical Metrics (ROC, Specificity, AUROC)
2:45 - 3:00  |  Limitations & Future Roadmap
```

---

## Detailed Section-by-Section Script

### 1. Hook & Problem Statement (0:00 – 0:30)
- **Visual**: Title slide with project name, followed by statistics/visuals of dermatological screening delays.
- **Voiceover**:
  > "Skin cancer is one of the most widespread cancers worldwide, where timely intervention directly dictates patient survival. However, access to specialized dermatologists is heavily constrained, leading to critical screening bottlenecks.
  > While machine learning can automate triage, standard deep learning models operate as opaque 'black boxes'—returning a cold prediction without explaining *why*.
  > Today, we present the **Skin Lesion Triage Tool**: an explainable AI system that flags high-risk lesions while visually highlighting the diagnostic features driving its assessment."

---

### 2. Architecture & Technical Pipeline (0:30 – 1:15)
- **Visual**: High-level system architecture diagram (Data -> Grouped Split -> EfficientNet-B0 -> BCE Loss with Pos-Weight -> Grad-CAM).
- **Voiceover**:
  > "To build a robust and clinically sound pipeline:
  > - **Data & Leakage Prevention**: We utilize the HAM10000 dataset comprising 10,015 dermoscopy images across 7 diagnostic categories, mapped to binary triage risk. We implement a patient-stratified split using `StratifiedGroupKFold`, ensuring that images from the same patient never leak across training, validation, and test sets.
  > - **Model Backbone**: We leverage an ImageNet-pretrained **EfficientNet-B0** backbone optimized with class-weighted binary cross-entropy loss (`pos_weight`) to handle the natural 80/20 class imbalance.
  > - **Interpretability**: Crucially, we integrate **Grad-CAM** (Gradient-weighted Class Activation Mapping) on the final convolutional feature maps to extract spatial activation heatmaps for every prediction."

---

### 3. Live Demo Walkthrough (1:15 – 2:25)
- **Visual**: Screen recording of the live Streamlit Web App (`streamlit run src/app.py`).
- **Demo Actions & Voiceover**:
  - *Action 1: Load a Malignant Case (e.g., Melanoma / BCC)*
    > "Let's see the application in action. A clinician or healthcare worker loads a dermoscopy image into our interactive interface.
    > The model evaluates the lesion and immediately flags it as **Malignant / High Suspicion** with calibrated probability.
    > More importantly, look at the **Grad-CAM heatmap**: the red and yellow focus points concentrate directly on the irregular border and pigment network of the lesion—allowing clinicians to verify that the model is attending to genuine pathology rather than image artifacts."
  - *Action 2: Load a Benign Case (e.g., Melanocytic Nevus)*
    > "Next, testing a benign melanocytic nevus: the model assesses it as **Benign / Low Suspicion**, showing uniform low activation across the central pigment cluster, providing reassuring transparency."

---

### 4. Experimental Results & Metrics (2:25 – 2:45)
- **Visual**: Test ROC Curve, Confusion Matrix, and Reliability Diagram charts.
- **Voiceover**:
  > "On our held-out test split of 1,520 unseen patient images, the model achieves an **AUROC of 0.911** using test-time augmentation and temperature calibration.
  > At a high-specificity operating threshold of 95% specificity it maintains a sensitivity of **0.59** — and at the triage operating point of 90% specificity it catches **89% of malignant lesions** while correctly clearing three-quarters of benign cases, which is exactly the behavior a screening aid needs.
  > The reliability diagram confirms that the reported probabilities are now well calibrated, so clinicians can trust the confidence scores."

---

### 5. Ethical Considerations, Limitations & Next Steps (2:45 – 3:00)
- **Visual**: Conclusion slide with GitHub repository link and future milestones.
- **Voiceover**:
  > "As a research demonstration, our model acknowledges known demographic biases in public datasets and requires multi-center validation before clinical deployment.
  > We have already prototyped a multimodal variant that fuses clinical metadata — patient age, sex, and lesion site — reaching an AUROC of 0.905, and we plan to expand this on larger, more diverse cohorts like ISIC 2024.
  > Thank you for watching!"

---

## Recording Checklist for the Presenter

- [ ] Record in 1080p or 4K resolution (16:9 aspect ratio).
- [ ] Clear microphone audio without background noise.
- [ ] Upload video to YouTube as **Unlisted** (Public is also acceptable; Private will not be accepted by Devpost).
- [ ] Verify the link is accessible in an incognito window.
- [ ] Paste the YouTube link into the Devpost submission form.
