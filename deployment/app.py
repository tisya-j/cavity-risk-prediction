import os
import joblib
import pandas as pd
import gradio as gr

# --- 1. DYNAMIC FILE PATH FIX ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cavity_model.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")

# Load assets safely
model = joblib.load(model_path)
features = joblib.load(features_path)


# --- 2. PREDICTION FUNCTION ---
def predict(
    age,
    income,
    calories,
    carbs,
    sugar,
    bmi,
    waist,
    systolic,
    diastolic,
    cholesterol,
    poor_oral_health,
    smoker,
    infrequent_dental_visit,
):
    # Core mathematical/behavioral transformations
    oral_risk = (
        int(poor_oral_health)
        + int(smoker)
        + int(infrequent_dental_visit)
    )

    waist_bmi_ratio = waist / (bmi + 1)

    X = pd.DataFrame([{
        "RIDAGEYR": age,
        "INDHHIN2": income,
        "DR1TKCAL": calories,
        "DR1TCARB": carbs,
        "DR1TSUGR": sugar,
        "BMXBMI": bmi,
        "BMXWAIST": waist,
        "BPXSY1": systolic,
        "BPXDI1": diastolic,
        "LBXTC": cholesterol,
        "waist_bmi_ratio": waist_bmi_ratio,
        "oral_risk": oral_risk
    }])

    # Align columns exactly to model expectations
    X = X[features]

    # Predict risk metrics
    probability = model.predict_proba(X)[0][1]

    if probability >= 0.7:
        level = "High Risk"
    elif probability >= 0.4:
        level = "Moderate Risk"
    else:
        level = "Low Risk"

    return (
        f"Predicted cavity risk: {level}\n\n"
        f"Probability: {probability:.1%}"
    )


# --- 3. REFINED BLOCKS INTERFACE WITH UNITS ---
with gr.Blocks(title="NHANES Cavity Risk Predictor") as demo:
    gr.Markdown("# NHANES Cavity Risk Predictor")
    gr.Markdown("Predict cavity risk using demographic, dietary, metabolic, and oral health indicators.")
    
    with gr.Row():
        # Left Panel: Categorized User Inputs
        with gr.Column(scale=2):
            
            with gr.Group():
                gr.Markdown("### Demographics")
                age = gr.Slider(1, 100, value=30, label="Age (years)")
                income = gr.Number(value=5, label="Income Category Index (INDHHIN2)")
                
            with gr.Group():
                gr.Markdown("### Dietary Metrics")
                with gr.Row():
                    calories = gr.Number(value=1700, label="Daily Energy Intake (kcal)")
                    carbs = gr.Number(value=250, label="Daily Carbohydrates (g)")
                    sugar = gr.Number(value=50, label="Daily Sugar Intake (g)")
                    
            with gr.Group():
                gr.Markdown("### Metabolic Indicators and Vitals")
                with gr.Row():
                    bmi = gr.Number(value=22.0, label="Body Mass Index (kg/m²)")
                    waist = gr.Number(value=80.0, label="Waist Circumference (cm)")
                    cholesterol = gr.Number(value=180, label="Total Cholesterol (mg/dL)")
                with gr.Row():
                    systolic = gr.Number(value=120, label="Systolic Blood Pressure (mmHg)")
                    diastolic = gr.Number(value=80, label="Diastolic Blood Pressure (mmHg)")
                    
            with gr.Group():
                gr.Markdown("### Oral Health and Behavioral Factors")
                with gr.Row():
                    poor_oral_health = gr.Checkbox(label="Poor Oral Health Symptoms")
                    smoker = gr.Checkbox(label="Tobacco Smoker")
                    infrequent_dental_visit = gr.Checkbox(label="Infrequent Dental Visits")
        
        # Right Panel: Output and Triggers
        with gr.Column(scale=1):
            gr.Markdown("### Assessment Output")
            output_box = gr.Textbox(label="Output Window", interactive=False, lines=5)
            
            submit_btn = gr.Button("Calculate Risk Profile", variant="primary")

    # Link interactive submit functionality to UI logic
    submit_btn.click(
        fn=predict,
        inputs=[
            age, income, calories, carbs, sugar, 
            bmi, waist, systolic, diastolic, cholesterol, 
            poor_oral_health, smoker, infrequent_dental_visit
        ],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()