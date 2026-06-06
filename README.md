# Dental Cavity Risk Prediction using NHANES Data

An end-to-end explainable machine learning system built to predict population-level dental cavity risk. This project bridges predictive modeling with clinical interpretability using SHAP-based feature attribution and hosts a localized, patient-centered screening interface.

**Live Deployment Interface:** [Interact with the Web Application on Hugging Face Spaces](https://huggingface.co/spaces/Tisya-J/cavity-risk-prediction)

---

## Project Architecture

```text
cavity-risk-ml/
├── .gitignore                    # System configuration to exclude temporary and cache files
├── README.md                     # Main project documentation and live deployment links
├── requirements.txt              # Core project-wide Python dependency specifications
│
├── data/
│   └── clean_nhanes.csv          # Preprocessed dataset containing custom engineered features
│
├── deployment/
│   ├── app.py                    # Formatted, multi-panel Gradio UI web hosting environment
│   ├── requirements.txt          # Cloud-specific container package dependencies
│   ├── cavity_model.pkl          # Serialized production XGBoost machine learning pipeline
│   └── features.pkl              # Expected feature order mapping serialization file
│
├── images/
│   ├── force_plot.png            # SHAP individual prediction force plot visualization
│   ├── shap_bar.png              # Mean absolute SHAP feature importance plot
│   └── shap_summary.png          # SHAP global impact distribution visualization
│
└── notebooks/
    └── NHAMEScavityrisk.ipynb    # Main model development, optimization, and evaluation notebook
```

---

## Problem Statement

Dental caries is a multifactorial condition influenced by overlapping demographic, behavioral, dietary, and socioeconomic variables. This project develops an optimized classifier capable of identifying individuals at risk of dental caries while providing transparent, patient-level feature attributions to support clinical screening workflows.

---

## Dataset & Feature Engineering

The underlying data is extracted from the National Health and Nutrition Examination Survey (NHANES). To transition this model into a user-friendly screening product, the deployment layer abstracts confusing data codes into intuitive consumer metrics:

* **Socioeconomic Localization:** Raw NHANES income indices (`INDHHIN2`) are abstracted into mapped annual household dropdown brackets localized in Indian Rupees (INR, ₹) and adjusted for Purchasing Price Parity (PPP)
* **Automated Biometrics:** Manual BMI input criteria are abstracted out. The application accepts everyday inputs (**Height in cm** and **Weight in kg**), calculating BMI automatically behind the scenes:
  $$\text{BMI} = \frac{\text{Weight (kg)}}{\left(\frac{\text{Height (cm)}}{100}\right)^2}$$
* **Engineered Oral Behavior Index:** Compiled internally as an aggregate integer (`oral_risk`) summing affirmative responses across poor self-reported health symptoms, nicotine consumption habits, and lack of dental checkups over the preceding 12 months.

---

###  Clinical Domain Validation & Professional Advisory
To ensure statistical robustness and practical clinical utility, the feature engineering pipeline and regional localization criteria were reviewed and audited in direct consultation with **practicing dental professionals**. 

Key strategic advice incorporated from medical experts includes:
1. **Behavioral Risk Compounding:** Validated that separate lifestyle metrics (tobacco use, self-reported localized pain, and avoidance of dental scale-and-clean checkups) should be consolidated into a unified behavioral index (`oral_risk`) to model how compounding bad habits exponentially raise the rate of cariogenic tooth decay.
2. **Socioeconomic Baseline Calibration:** Advised that raw income metrics from a US public health registry would misrepresent localized baseline vulnerabilities without accounting for geographical purchasing indices. Implementing the **PPP adjustment layer** successfully calibrated the socioeconomic risk signals to accurately reflect real-world local contexts.
3. **Actionable Clinical Translation:** Advised transitioning from technical, raw data indices to transparent reference metrics (World Health Organization free sugar benchmarks, clear systemic vital targets) on the UI to guide high-risk screening candidates toward meaningful, preventive checkups.


## Model Evaluation

Models were optimized using a stratified train-test split configuration to strictly isolate and evaluate target class boundaries.


| Metric | Baseline Logistic Regression | Production XGBoost Classifier |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.74 | **0.85** |
| **Accuracy** | 0.69 | **0.78** |
| **F1 Score** | 0.71 | **0.82** |

---

## Interpretability & SHAP Analysis

Global model decisions were audited using SHAP feature attribution to ensure strict compliance with established public health literature.

### Global Attributions
* **Age (RIDAGEYR):** Identified as the strongest predictor, mapping cumulative lifetime exposure vector patterns.
* **Adiposity Measures (BMXWAIST):** Higher waist circumference structures positive correlation steps toward risk amplification.
* **Dietary Variables:** Daily sugar and caloric indicators add consistent positive risk deltas.
* **Socioeconomic Status:** Low household income indices register a measurable structural impact on target prevalence rates.

---

## Production Deployment

The final pipeline is bundled and deployed using an interactive Python layout:
* **Backend Architecture:** Production model weights are dynamic-pathed and evaluated over an XGBoost instance inside `app.py`.
* **Frontend UI:** Built via Gradio blocks partitioned into explicit, reader-friendly sections detailing health reference guidelines (WHO sugar limits, blood pressure thresholds) and model validation metrics.
* **Cloud Hosting:** Hosted in an isolated cloud virtualization sandbox on Hugging Face Spaces.

---

## Technologies Used

* **Core Engines:** Python, Pandas, NumPy, Joblib
* **Modeling & Optimization:** Scikit-learn, XGBoost
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Interface Layer:** Gradio, Markdown
