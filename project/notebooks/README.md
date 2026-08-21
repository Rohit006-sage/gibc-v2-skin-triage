# Kaggle Notebook Setup (Step-by-step)

This file documents how to get training running on Kaggle Notebooks in
~10 minutes. If you prefer to just run code, paste the contents of
`kaggle_setup.py` into a single notebook cell instead.

## Prerequisites (do these once)

1. **Create a GitHub repo** with the contents of `project/` (minus
   `data/raw/` — that's gitignored). Note the URL; you'll paste it
   into the setup cell.

2. **Create a Kaggle account** at https://www.kaggle.com (free).

3. **Get a Kaggle API token** for the setup script to download data:
   - Go to https://www.kaggle.com/settings
   - Click "Create New Token"
   - This downloads `kaggle.json`
   - In your notebook environment, when you click "Add secret" in the
     Secrets panel, add `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Steps

### 1. Create a new notebook

Go to https://www.kaggle.com/code → click **"+ New Notebook"**.

### 2. Enable GPU

Right sidebar → **Accelerator** → **GPU T4 ×2** (free tier).

### 3. Enable Kaggle API access

Right sidebar → under "Variables", add secrets:
- `KAGGLE_USERNAME` = your Kaggle username
- `KAGGLE_KEY` = your kaggle.json key value

Then in the first cell:

```python
import os
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_kaggle_api_key"
```

### 4. Edit the constants in `kaggle_setup.py`

Open `notebooks/kaggle_setup.py` and replace:
- `GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"`
  with your actual repo URL.

If your repo is **private**, also paste a GitHub personal access token
into the Secrets panel as `GITHUB_TOKEN`, and the script will use it.

### 5. Paste the setup cell

In your notebook, add a new cell and paste the **entire contents** of
`notebooks/kaggle_setup.py`. Run it.

You should see:
```
PyTorch: 2.x.x
CUDA available: True
Device: Tesla T4
...
[OK] /kaggle/working/project/data/raw/HAM10000_metadata.csv
[OK] /kaggle/working/project/data/raw/HAM10000_images_part_1
[OK] /kaggle/working/project/data/raw/HAM10000_images_part_2
[OK] (a) no patient ID appears in more than one split
[OK] (b) class distribution preserved within 3pp across splits
[OK] (c) sample loads with shape (3, 450, 600), label=0, id=ISIC_0027419
```

### 6. Train

In a new cell:

```python
!cd /kaggle/working/project && python -m src.train \
    --metadata data/raw/HAM10000_metadata.csv \
    --image-dir data/raw/HAM10000_images_part_1 \
    --image-dir data/raw/HAM10000_images_part_2 \
    --output-dir runs/baseline \
    --arch efficientnet_b0 \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4
```

### 7. Save outputs back to your repo (so you don't lose them)

After training, in a new cell:

```python
import subprocess
subprocess.run(
    "git add runs/baseline/ training_log.csv && "
    "git commit -m 'baseline run: val_auc=X.XX' && "
    "git push",
    shell=True, check=True, cwd="/kaggle/working/project",
)
```

You can also download `best.pt` directly from the notebook's output
panel for use in the Streamlit app locally.

## What's next after training finishes

- Open `runs/baseline/training_log.csv` to see the per-epoch metrics.
- Run `python -m src.evaluate --checkpoint runs/baseline/best.pt ...`
  to get final test-set numbers.
- Paste the test AUROC + sens@95%spec back to the AI agent and ask for
  the next iteration (week 3 plan).
