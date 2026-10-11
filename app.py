"""Fast Food Calorie Shock — workout burn calculator (Gradio)."""
import os
import joblib
import pandas as pd
import gradio as gr

reg = joblib.load("burn_time_model.joblib")     # predicts walking_burn_time_min
clf = joblib.load("shock_tier_model.joblib")    # predicts workout_shock_tier
# optional: calibrated 80% prediction interval {"q10": model, "q90": model, "adj": float}
INTERVAL = joblib.load("burn_time_interval.joblib") if os.path.exists("burn_time_interval.joblib") else None

CAT_NAMES = ["fast_food_chain", "meal_category", "consumer_gender", "fitness_level"]
CATS = dict(zip(CAT_NAMES, reg.named_steps["pre"].named_transformers_["cat"].categories_))

# Average grams (mg for sodium) per calorie in the dataset -> used when macros are left at 0
PER_CAL = {"fat_grams": 0.0500, "saturated_fat_grams": 0.0200, "carb_grams": 0.1036,
           "sugar_grams": 0.0388, "protein_grams": 0.0337, "sodium_mg": 1.497}
# Approximate scaling to other workouts (ratio of implied METs found in the EDA)
OTHER = {"Cycling": 3.5 / 7.4, "Running": 3.5 / 10.3, "HIIT": 3.5 / 12.1}


def predict(calories, weight, age, gender, fitness, chain, meal, items,
            fat, sat, carb, sugar, protein, sodium):
    vals = {"fat_grams": fat, "saturated_fat_grams": sat, "carb_grams": carb,
            "sugar_grams": sugar, "protein_grams": protein, "sodium_mg": sodium}
    row = {k: (v if v else PER_CAL[k] * calories) for k, v in vals.items()}
    row.update({"item_count": items, "total_calories": calories,
                "consumer_age": age, "consumer_weight_kg": weight,
                "consumer_gender": gender, "fitness_level": fitness,
                "fast_food_chain": chain, "meal_category": meal})
    row["cal_per_kg"] = calories / weight                      # engineered feature used by the regression model
    full = pd.DataFrame([row])
    cols = lambda model: list(model.named_steps["pre"].feature_names_in_)   # each model gets exactly its own columns
    X = full[cols(reg)]

    walk = float(reg.predict(X)[0])
    probs = clf.predict_proba(full[cols(clf)])[0]
    classes = clf.named_steps["model"].classes_
    tier = classes[probs.argmax()]

    lines = [f"## 🚶 {walk:.0f} min of walking  (≈ {walk/60:.1f} h)"]
    if INTERVAL:
        a, b = float(INTERVAL["q10"].predict(full[cols(INTERVAL["q10"])])[0]), float(INTERVAL["q90"].predict(full[cols(INTERVAL["q90"])])[0])
        lo = max(0.0, min(a, b, walk) - INTERVAL["adj"]); hi = max(a, b, walk) + INTERVAL["adj"]
        lines += [f"**Likely range (80%): {lo:.0f}–{hi:.0f} min**"]
    lines += [f"- {n}: ~{walk*r:.0f} min" for n, r in OTHER.items()]
    lines += [f"\n**Shock tier:** {tier}"]
    if not 750 <= calories <= 3600:
        lines += ["\n⚠️ Calories are outside the range the model was trained on (750–3,600 kcal)."]
    lines += ["\n*Cycling/running/HIIT are approximations scaled from the walking prediction.*"]
    return "\n".join(lines), {c: float(p) for c, p in zip(classes, probs)}


with gr.Blocks(title="Fast Food Calorie Shock") as demo:
    gr.Markdown("# 🍔🔥 Fast Food Calorie Shock\nHow long must you exercise to burn off your order?")
    with gr.Row():
        with gr.Column():
            calories = gr.Slider(300, 4000, value=1500, step=10, label="Total calories (kcal)")
            weight = gr.Slider(40, 150, value=75, step=0.5, label="Body weight (kg)")
            age = gr.Slider(18, 80, value=35, step=1, label="Age")
            gender = gr.Dropdown(CATS["consumer_gender"].tolist(), value=CATS["consumer_gender"][0], label="Gender")
            fitness = gr.Dropdown(CATS["fitness_level"].tolist(), value=CATS["fitness_level"][0], label="Fitness level")
            chain = gr.Dropdown(CATS["fast_food_chain"].tolist(), value=CATS["fast_food_chain"][0], label="Chain")
            meal = gr.Dropdown(CATS["meal_category"].tolist(), value=CATS["meal_category"][0], label="Meal category")
            items = gr.Slider(1, 8, value=3, step=1, label="Item count")
            with gr.Accordion("Advanced: macros (0 = estimate from calories)", open=False):
                fat = gr.Number(value=0, label="Fat (g)")
                sat = gr.Number(value=0, label="Saturated fat (g)")
                carb = gr.Number(value=0, label="Carbs (g)")
                sugar = gr.Number(value=0, label="Sugar (g)")
                protein = gr.Number(value=0, label="Protein (g)")
                sodium = gr.Number(value=0, label="Sodium (mg)")
            btn = gr.Button("Calculate", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_lbl = gr.Label(label="Shock tier probabilities")
    btn.click(predict, [calories, weight, age, gender, fitness, chain, meal, items,
                        fat, sat, carb, sugar, protein, sodium], [out_md, out_lbl])

if __name__ == "__main__":
    demo.launch()
