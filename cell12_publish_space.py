# Cell 12 – Publish the design aid as a permanent public web app + API (free Hugging Face Space)
#   Run after Cell 10. Before running:
#     1. Create a free account at https://huggingface.co (if you don't have one).
#     2. Settings -> Access Tokens -> "Create new token" with WRITE permission.
#     3. Change SPACE below to "<your-username>/<app-name>".
#   login() asks for the token in a box; the token is not written into the notebook.
#   Everything the app needs (model members, aleatoric model, bundle.json) is uploaded with it,
#   so users do NOT need your dataset or notebook.
import os, shutil, gradio, xgboost
from huggingface_hub import login, HfApi

SPACE = "your-username/uhpc-joint-design-aid"          # <-- change this
PRIVATE = False                                         # True = only you (and people you invite) can open it

login()
SPACE_DIR = f"{OUT}/hf_space"
shutil.rmtree(SPACE_DIR, ignore_errors=True)
shutil.copytree(APP_DIR, SPACE_DIR, ignore=shutil.ignore_patterns(
    "review_queue.csv", "reviewed_cases_export.csv", "figures", "__pycache__", "*.log", "*.ipynb"))
for f in ("app.py", "html-to-image.js"):                # the latest versions next to the notebook
    shutil.copy(f, SPACE_DIR)

open(f"{SPACE_DIR}/requirements.txt", "w").write(
    f"xgboost=={xgboost.__version__}\nnumpy\npandas\nscipy\nmatplotlib\n")
open(f"{SPACE_DIR}/packages.txt", "w").write("fonts-liberation\n")   # Times-compatible font for the plots
open(f"{SPACE_DIR}/README.md", "w", encoding="utf-8").write(f"""---
title: UHPC joint shear capacity design aid
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: {gradio.__version__}
app_file: app.py
pinned: false
---
Human-AI collaborative design aid for the shear capacity of match-cast UHPC joints
(physics-constrained XGB ensemble with 95% conformal intervals and uncertainty-gated manual inspection).

API: `from gradio_client import Client; Client("{SPACE}").predict("Dry", "Single", 3, 44, 150, 100, 200, 800,
14, 149, 11, 2, api_name="/predict")`
""")

api = HfApi()
api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=PRIVATE, exist_ok=True)
api.upload_folder(folder_path=SPACE_DIR, repo_id=SPACE, repo_type="space",
                  commit_message="Publish UHPC joint design aid")
user, name = SPACE.split("/")
print(f"Uploaded. The Space builds in 2-5 minutes, then:")
print(f"  Web app : https://huggingface.co/spaces/{SPACE}")
print(f"  Direct  : https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")
print(f"  API     : Client('{SPACE}').predict(..., api_name='/predict')")
