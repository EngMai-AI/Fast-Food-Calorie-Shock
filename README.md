🍔🔥 Fast Food Calorie Shock — EDA, Machine Learning & Dashboard

How long do you have to exercise to burn off a fast-food order? An end-to-end data project on the Kaggle dataset Fast Food Calorie Shock: Food vs Workout Burn (8,000 orders × 23 columns): data audit, EDA, regression and classification, calibrated prediction intervals, an interactive calculator and a Power BI dashboard.

🔗 Live demo: <add your Hugging Face Space link> · 📓 Kaggle notebook: <add your Kaggle link>

Headline finding

For the same 1,000–1,200 kcal order, the lightest quarter of consumers needs a median of 303 walking minutes, the heaviest quarter only 195 (−36%). Burn time depends on calories ÷ body weight and on fitness level — not on the chain or the macronutrients.

Key results
Task	Model	Result
Walking minutes (regression)	Gradient Boosting + cal_per_kg	MAE 3.9 min, R² 0.999
Baselines	Textbook physics (MET 3.5) / physics + MET per fitness level	MAE 17.7 / 4.3 min
workout_shock_tier (4 classes)	Logistic Regression	93.6% accuracy, macro-F1 0.86
Uncertainty	Quantile GB + conformal calibration	80.5% coverage of an 80% interval
What I found along the way
total_calories is clipped at 750 (8.8% of orders) and 3,600 (1.5%) → synthetic-looking data.
Back-solved MET from calories: walking ≈ 3.5, cycling ≈ 7.4, running ≈ 10.3, HIIT ≈ 12.1; walking MET by fitness level 3.29 / 3.57 / 3.86.
A 3-parameter physics formula (MAE 4.3) nearly matches the best ML model (MAE 3.9): the value is the formula discovery and the cal_per_kg feature.
workout_shock_tier is defined by running minutes (cut-offs at 45, 80 and 120).
Leakage experiment: adding derived/burn-time columns lifts tier accuracy from 93.6% to 100% — a score nobody could reproduce before exercising.
Highly Trained consumers need ~15% fewer minutes than Sedentary ones; gender and age show no effect.
Repository structure
notebooks/   fast_food_calorie_shock_final.ipynb   # full analysis with outputs
app/         app.py, requirements.txt, *.joblib    # Gradio calculator (saved models)
dashboard/   *.pbix, screenshots/                  # Power BI dashboard
Run the app locally
bash
cd app
pip install -r requirements.txt     # scikit-learn pinned to the version used for training
python app.py
Data

The dataset is not included — download it from the Kaggle page and see the license there.

Limitations

Synthetic-looking data with clipped calories, MET values are population averages, a single train/test split plus 5-fold CV, and interval coverage is guaranteed on average, not per subgroup.
