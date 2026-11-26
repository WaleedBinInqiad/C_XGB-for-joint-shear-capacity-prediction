# Cell 12 – Prepare a GitHub repository for the design aid (no dataset needed by users)
#   Run after Cell 10. It builds results/github_repo/ (and a .zip of it) containing:
#     app.py, html-to-image.js          the design aid
#     bundle.json, aleatoric.json,      the trained C_XGB model exported by Cell 10
#     members/                          (no experimental data table is included)
#     requirements.txt, README.md, .gitignore, run_in_colab.ipynb
#   Then upload the folder's contents to a new GitHub repository (see the README text below).
import os, json, shutil, gradio, xgboost

GITHUB_USER = "your-username"            # <-- change
REPO = "uhpc-joint-design-aid"           # <-- change if you like

BUNDLE = json.load(open(f"{APP_DIR}/bundle.json"))
REPO_DIR = f"{OUT}/github_repo"
shutil.rmtree(REPO_DIR, ignore_errors=True)
shutil.copytree(APP_DIR, REPO_DIR, ignore=shutil.ignore_patterns(
    "review_queue.csv", "reviewed_cases_export.csv", "figures", "__pycache__", "*.log", "*.ipynb",
    "requirements.txt"))
for f in ("app.py", "html-to-image.js"):
    shutil.copy(f, REPO_DIR)

open(f"{REPO_DIR}/requirements.txt", "w").write(
    f"gradio=={gradio.__version__}\nxgboost=={xgboost.__version__}\nnumpy\npandas\nscipy\nmatplotlib\n")
open(f"{REPO_DIR}/.gitignore", "w").write("review_queue.csv\nreviewed_cases_export.csv\nfigures/\n__pycache__/\n")

colab = f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO}/blob/main/run_in_colab.ipynb"
open(f"{REPO_DIR}/README.md", "w", encoding="utf-8").write(f"""# Shear capacity of match-cast UHPC joints – human–AI design aid

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})

Design aid for the shear capacity of match-cast (dry and epoxy) UHPC joints. A physics-constrained XGBoost
ensemble (C_XGB) predicts the capacity with a calibrated 95% prediction interval and a SHAP explanation.
A prediction is released only if (G1) the model is at least 95% confident that the true capacity lies within
±20% of it, (G2) all inputs lie within the tested range and (G3) the joint is similar to a tested specimen;
otherwise it is passed to an engineer for manual inspection. The trained model is included, so the
experimental database is not needed to run it.

## Run without installing anything
Click **Open in Colab** above, then *Runtime → Run all*. A link to the design aid appears at the bottom
(valid while the Colab session is open).

## Run on your computer
Requires Python 3.10 or later.
```bash
git clone https://github.com/{GITHUB_USER}/{REPO}.git
cd {REPO}
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:7860 in a browser. Add `--share` for a temporary public link.

## Use from Python (API)
With the app running:
```python
from gradio_client import Client
c = Client("http://127.0.0.1:7860/")
c.predict("Dry", "Single", 3, 44, 150, 100, 200, 800, 14, 149, 11, 2, api_name="/predict")
```
Inputs, in order: interface (Dry/Epoxy), test configuration (Single/Double), number of keys (0 = flat),
key depth, key width, key height, joint width, shear-plane height (mm), confining stress, compressive strength,
tensile strength (MPa; 0 = estimate as 0.648√fc), fibre content (%).
Output: predicted capacity, 95% interval, confidence within ±20%, data/model uncertainty split, and whether
manual inspection is required, with the reasons.

## Model
Trained on {BUNDLE['n_specimens']} specimens from {BUNDLE['n_studies']} studies; xgboost {xgboost.__version__}.
Predictions are mean values without partial safety factors and do not replace code checks.

## Citation
If you use this design aid, please cite: [paper reference]
""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"colab": {"name": "run_in_colab.ipynb"}},
      "cells": [
          {"cell_type": "markdown", "metadata": {}, "source": ["# UHPC joint design aid\n",
                                                             "Runtime → Run all. The app link appears below."]},
          {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
              f"!git clone -q https://github.com/{GITHUB_USER}/{REPO}.git\n",
              f"%cd {REPO}\n",
              "!pip install -q -r requirements.txt"]},
          {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
              "import app\n", "app.launch(share=True)"]}]}
json.dump(nb, open(f"{REPO_DIR}/run_in_colab.ipynb", "w"), indent=1)

shutil.make_archive(REPO_DIR, "zip", REPO_DIR)
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(REPO_DIR) for f in fs) / 1e6
print(f"Repository folder: {REPO_DIR}  ({size:.1f} MB)  and  {REPO_DIR}.zip")
print("Contents:", sorted(os.listdir(REPO_DIR)))
