import os
import joblib
import pandas as pd
import gradio as gr

# --- 1. FILE SYSTEM INITIALIZATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cavity_model.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)

# NHANES INDHHIN2 categorical mapping dictionary converted to INR
INCOME_MAPPING = {
    "₹0 to ₹4,25,000": 1,
    "₹4,25,000 to ₹8,50,000": 2,
    "₹8,50,000 to ₹12,75,000": 3,
    "₹12,75,000 to ₹17,00,000": 4,
    "₹17,00,000 to ₹21,25,000": 5,
    "₹21,25,000 to ₹29,75,000": 6,
    "₹29,75,000 to ₹38,25,000": 7,
    "₹38,25,000 to ₹46,75,000": 8,
    "₹46,75,000 to ₹55,25,000": 9,
    "₹55,000,000 to ₹63,75,000": 10,
    "₹63,75,000 to ₹85,00,000": 12,
    "₹85,00,000 and Over": 15,
}

# --- 2. EVALUATION & PREDICTION CORE ---
def predict(
    age,
    income_bracket,
    calories,
    carbs,
    sugar,
    weight,
    height,
    waist,
    systolic,
    diastolic,
    cholesterol,
    poor_oral_health,
    smoker,
    infrequent_dental_visit,
):
    income_code = INCOME_MAPPING.get(income_bracket, 5)

    # Automated BMI computation
    height_m = height / 100
    computed_bmi = weight / (height_m ** 2)

    # Compute custom feature extraction targets
    oral_risk = int(poor_oral_health) + int(smoker) + int(infrequent_dental_visit)
    waist_bmi_ratio = waist / (computed_bmi + 1)

    X = pd.DataFrame([{
        "RIDAGEYR": age,
        "INDHHIN2": income_code,
        "DR1TKCAL": calories,
        "DR1TCARB": carbs,
        "DR1TSUGR": sugar,
        "BMXBMI": computed_bmi,
        "BMXWAIST": waist,
        "BPXSY1": systolic,
        "BPXDI1": diastolic,
        "LBXTC": cholesterol,
        "waist_bmi_ratio": waist_bmi_ratio,
        "oral_risk": oral_risk
    }])

    X = X[features]
    probability = model.predict_proba(X)[0][1]

    if probability >= 0.7:
        level = "High Risk"
    elif probability >= 0.4:
        level = "Moderate Risk"
    else:
        level = "Low Risk"

    return (
        f"Predicted cavity risk: {level}\n\n"
        f"Probability: {probability:.1%}\n"
        f"Calculated Body Mass Index: {computed_bmi:.1f} kg/m²"
    )

# --- 3. PRISTINE Standard Interface ---
# Using the stable default framework setup to clear out glitches
with gr.Blocks(theme=gr.themes.Default(), title="NHANES Cavity Risk Predictor") as demo:
    gr.Markdown("# NHANES Cavity Risk Predictor")
    gr.Markdown("An accessible, patient-centered screening application powered by machine learning.")
    
    with gr.Row():
        with gr.Column(scale=2):
            
            with gr.Group():
                gr.Markdown("### Demographics & Socioeconomic Status")
                age = gr.Slider(1, 100, value=30, label="Age (years)")
                income_bracket = gr.Dropdown(
                    choices=list(INCOME_MAPPING.keys()), 
                    value="₹38,25,000 to ₹46,75,000", 
                    label="Annual Household Income"
                )
                
            with gr.Group():
                gr.Markdown("### Dietary Metrics (Estimated Daily Consumptions)")
                gr.Markdown("*Tip: A standard adult profile averages ~2000 kcal, 250g carbs, and under 50g sugar.*")
                with gr.Row():
                    calories = gr.Slider(500, 5000, value=1800, step=50, label="Daily Calories (kcal)")
                    carbs = gr.Slider(50, 600, value=250, step=10, label="Daily Carbohydrates (g)")
                    sugar = gr.Slider(0, 300, value=45, step=5, label="Daily Sugar Intake (g)")
                    
            with gr.Group():
                gr.Markdown("### Biometrics & Vital Signs")
                with gr.Row():
                    height = gr.Number(value=170, label="Height (cm)")
                    weight = gr.Number(value=70, label="Weight (kg)")
                    waist = gr.Number(value=85, label="Waist Circumference (cm)")
                with gr.Row():
                    cholesterol = gr.Number(value=180, label="Total Cholesterol (mg/dL)")
                    systolic = gr.Number(value=120, label="Systolic Blood Pressure (mmHg)")
                    diastolic = gr.Number(value=80, label="Diastolic Blood Pressure (mmHg)")
                    
            with gr.Group():
                gr.Markdown("### Oral Health Habits")
                poor_oral_health = gr.Checkbox(label="Experiencing toothache or oral discomfort")
                smoker = gr.Checkbox(label="Frequent tobacco or nicotine consumption")
                infrequent_dental_visit = gr.Checkbox(label="No dental visit within the past year")
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Assessment Output")
                output_box = gr.Textbox(label="Output Window", interactive=False, lines=5)
                submit_btn = gr.Button("Calculate Risk Profile", variant="primary")
            
            with gr.Group():
                gr.Markdown("### Model Performance Metrics")
                gr.Markdown(
                    "The validation statistics for the underlying production XGBoost model are as follows:\n\n"
                    "* **ROC-AUC:** ~0.85\n"
                    "* **Accuracy:** ~0.78\n"
                    "* **F1-Score:** ~0.82"
                )
            
            with gr.Group():
                gr.Markdown("### About the Dataset")
                gr.Markdown(
                    "This model is trained on data from the **National Health and Nutrition Examination Survey (NHANES)**. "
                    "This cross-sectional program assesses the health and nutritional status of adults and children."
                )
            
            with gr.Group():
                gr.Markdown("### Health Metric Reference Ranges")
                gr.Markdown(
                    "**Sugar Intake:** Keep free sugar intake below **50 grams per day** to lower risk.\n\n"
                    "**Body Mass Index (BMI):** Calculated from weight and height. **18.5 – 24.9 kg/m²** is normal.\n\n"
                    "**Blood Pressure:** Baseline values should sit below **120 mmHg (systolic)** and **80 mmHg (diastolic)**.\n\n"
                    "**Total Cholesterol:** Target levels track below **200 mg/dL**."
                )

    submit_btn.click(
        fn=predict,
        inputs=[
            age, income_bracket, calories, carbs, sugar, 
            weight, height, waist, systolic, diastolic, cholesterol, 
            poor_oral_health, smoker, infrequent_dental_visit
        ],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch(js='() => { document.body.classList.remove("dark"); }')
