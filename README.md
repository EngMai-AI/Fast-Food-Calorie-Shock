---
title: Fast Food Calorie Shock
emoji: 🍔
colorFrom: red
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# 🍔🔥 Fast Food Calorie Shock — How long to burn it off?

**Headline finding:** the same 1,000-kcal order needs roughly twice as many walking minutes for a 50 kg person as for a 100 kg person — burn time depends almost entirely on calories ÷ body weight, not on the chain or the macros.

An end-to-end project on the Kaggle dataset *Fast Food Calorie Shock: Food vs Workout Burn* (8,000 orders × 23 columns): EDA, data-quality audit, leakage handling, regression and classification, and an interactive calculator.

## Key results
| Task | Best model | Result |
|---|---|---|
| Walking minutes (regression) | Gradient Boosting + `cal_per_kg` | MAE 3.9 min, R² 0.999 (textbook physics: MAE 17.7; physics + fitness-level MET: MAE 4.3) |
| `workout_shock_tier` (4 classes) | Logistic Regression | 93.6% accuracy, macro-F1 0.86 |

## What I found along the way
- `total_calories` is clipped at 750 (8.8% of orders) and 3,600 (1.5%) → synthetic-looking data.
- Back-solved MET from `total_calories`: walking ≈ 3.5 → burn time is largely `calories / (MET × 0.0175 × weight)`.
- Chains and meal categories are nearly identical (1,575–1,622 kcal avg) and carry ~0 importance.
- Leaky columns (other burn times, derived calorie features, the tier itself for regression) were excluded; a leakage experiment shows how much they inflate the scores.
- Predictions come with a calibrated 80% interval (conformal quantile regression), not just a point estimate.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Put `burn_time_model.joblib`, `shock_tier_model.joblib` and `burn_time_interval.joblib` (saved at the end of the notebook) next to `app.py`. The interval file is optional: without it the app shows a single number.
`requirements.txt` pins `scikit-learn==1.6.1` (the version used on Kaggle when the models were saved). If you retrain elsewhere, pin the version printed by the last notebook cell.

## Deploy on Hugging Face Spaces
1. Create a new Space → SDK **Gradio**.
2. Upload `app.py`, `requirements.txt`, `README.md` and the three `.joblib` files.
3. Wait for the build, then add the Space link to your CV / GitHub.
