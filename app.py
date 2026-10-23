import gradio as gr
import pandas as pd
import joblib

# Load the trained pipeline
diabetes_model = joblib.load("models/diabetes_pipeline.joblib")

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age
):
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }])

    prediction = diabetes_model.predict(input_data)[0]
    probability = diabetes_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "🟠 The model predicts a positive outcome."
    else:
        result = "🟢 The model predicts a negative outcome."

    return (
        result,
        f"{probability:.1%}"
    )


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=1),
        gr.Number(label="Glucose", value=120),
        gr.Number(label="Blood Pressure", value=70),
        gr.Number(label="Skin Thickness", value=20),
        gr.Number(label="Insulin", value=79),
        gr.Number(label="BMI", value=25.0),
        gr.Number(label="Diabetes Pedigree Function", value=0.5),
        gr.Number(label="Age", value=30)
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Model Score for Positive Outcome")
    ],
    title="Diabetes Prediction Using Machine Learning",
    description=(
        "Enter the measurements to obtain an experimental model prediction. "
        "This tool is for educational purposes only and is not a medical "
        "diagnostic system."
    ),
    flagging_mode="never"
)

demo.launch()