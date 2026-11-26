# Shear capacity of match-cast UHPC joints – human–AI design aid

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaleedBinInqiad/uhpc-joint-design-aid/blob/main/run_in_colab.ipynb)

Design aid for the shear capacity of match-cast (dry and epoxy) UHPC joints in precast segmental bridges.
A physics-constrained XGBoost ensemble (C_XGB) predicts the shear capacity with a calibrated 95% prediction
interval and a SHAP explanation. A prediction is released automatically only if all release criteria are met:

- **G1** – the model is at least 95% confident that the true capacity lies within ±20% of the prediction;
- **G2** – every input lies within the range of the tested specimens;
- **G3** – the joint is similar to at least one tested specimen (nearest-neighbour distance ≤ Q3 + 1.5 IQR);
- **G4** – the engineer has not referred the case.

Otherwise, the prediction is passed for manual inspection, where the engineer accepts it, adopts its 95% lower
bound, overrides it, or requests a physical test, with a written justification.

The trained model is included, so the experimental database is **not** needed to run the design aid.

## Run without installing anything
Click **Open in Colab** above, then select *Runtime → Run all*. A link to the design aid appears at the bottom
of the notebook and stays active while the Colab session is open.

## Run on your computer
Requires Python 3.10 or later.

```bash
git clone https://github.com/WaleedBinInqiad/uhpc-joint-design-aid.git
cd uhpc-joint-design-aid
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860 in a web browser. Add `--share` to create a temporary public link.

## Using the design aid
1. Enter the joint geometry, loading and material properties, then click **Predict**.
2. The page shows the predicted shear capacity, the 95% prediction interval, the SHAP explanation and whether
   the prediction is passed for manual inspection.
3. For a flagged prediction, enter the reviewer, decision and justification, then click **Record decision**.
   Any prediction can also be referred with the **Manual inspection** button.
4. **Download page (PNG)** saves the page as an image.

## Use from Python (API)
With the app running:

```python
from gradio_client import Client
c = Client("http://127.0.0.1:7860/")
result = c.predict("Dry", "Single", 3, 44, 150, 100, 200, 800, 14, 149, 11, 2, api_name="/predict")
print(result)
```

| Order | Input | Unit / values |
|---|---|---|
| 1 | Interface | `Dry` or `Epoxy` |
| 2 | Test configuration | `Single` or `Double` |
| 3 | Number of keys | 0 = flat joint |
| 4 | Key depth | mm |
| 5 | Key width | mm |
| 6 | Key height | mm |
| 7 | Joint width | mm |
| 8 | Shear-plane height | mm |
| 9 | Confining stress | MPa |
| 10 | Compressive strength | MPa |
| 11 | Tensile strength | MPa (0 = estimated as 0.648√fc) |
| 12 | Fibre content | % |

The result gives the predicted capacity (kN), the 95% interval, the confidence within ±20%, the data/model
uncertainty split, and whether manual inspection is required, with the reasons.

## Repository contents
| File | Purpose |
|---|---|
| `app.py` | Design aid (interface and API) |
| `bundle.json` | Model settings, tested input ranges and interval calibration |
| `aleatoric.json` | Model of the data (aleatoric) uncertainty |
| `members/` | The 50 trained C_XGB ensemble members |
| `html-to-image.js` | Used by the page download button |
| `requirements.txt` | Python packages |
| `run_in_colab.ipynb` | Runs the design aid in Google Colab |

## Limitations
Predictions are mean values without partial safety factors and do not replace design-code checks. The model was
trained on 169 specimens from 16 studies; predictions outside the tested domain are flagged for manual
inspection.

## Citation
If you use this design aid, please cite:

[paper reference]
