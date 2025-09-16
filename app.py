# app.py (in the root directory)

import gradio as gr
from pathlib import Path
from huggingface_hub import snapshot_download
import asyncio
from PIL import Image

# --- Import and Initialize Backend Components from the 'app' folder ---
from app.prediction import PredictionPipeline
from app.database import add_patient_record, get_all_records

# Initialize components once
prediction_pipeline = PredictionPipeline()
HF_DATASET_REPO = "ALYYAN/chest-xray-pneumonia-samples"
try:
    SAMPLE_IMAGE_DIR = Path(snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset"))
    # Create a list of sample image paths for the Gradio component
    SAMPLE_IMAGES = [str(p) for p in list(SAMPLE_IMAGE_DIR.glob('*/*.jpeg'))[:10]]
except Exception as e:
    print(f"Could not download sample images: {e}")
    SAMPLE_IMAGES = []

# --- Core Prediction Logic for Gradio ---
async def classify_images(patient_name, patient_age, image_list):
    # 1. Input Validation
    if not patient_name or patient_age is None:
        raise gr.Error("Patient Name and Age are required.")
    if not image_list:
        raise gr.Error("Please upload at least one image.")
    
    # Gradio provides file paths for uploaded files in a temp directory
    # Our prediction pipeline can handle these paths directly.

    # 2. Run Prediction
    result = prediction_pipeline.predict(image_list) # Pass the list of temp file paths
    prediction = result.get("prediction", "Error")
    confidence = result.get("confidence", 0)

    if prediction == "Error":
        raise gr.Error(result.get("details", "An unknown error occurred during prediction."))

    # 3. Save to Database
    # Ensure age is an integer
    try:
        age = int(patient_age)
    except (ValueError, TypeError):
        raise gr.Error("Patient Age must be a valid number.")

    await add_patient_record(
        name=str(patient_name),
        age=age,
        result=prediction,
        confidence=confidence
    )

    # 4. Format the Output for Gradio
    confidences = {"NORMAL": 0.0, "PNEUMONIA": 0.0} # Initialize both labels
    confidences[prediction] = confidence
    
    return confidences

# --- Function to fetch and format database records ---
async def get_records_html():
    records = await get_all_records()
    if not records:
        return "<p>No records found in the database.</p>"
    
    # Create an HTML table from the records
    html = "<table><tr><th>Name</th><th>Age</th><th>Prediction</th><th>Confidence</th><th>Date</th></tr>"
    for r in records:
        confidence_percent = f"{r['confidence_score']:.2%}" if r['confidence_score'] is not None else "N/A"
        timestamp = r['timestamp'].strftime('%Y-%m-%d %H:%M') if r['timestamp'] else "N/A"
        html += f"<tr><td>{r.get('name', 'N/A')}</td><td>{r.get('age', 'N/A')}</td><td>{r.get('prediction_result', 'N/A')}</td><td>{confidence_percent}</td><td>{timestamp}</td></tr>"
    html += "</table>"
    return html

# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), css="table { width: 100%; border-collapse: collapse; } th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }") as demo:
    gr.Markdown("# 🩺 Pneumonia Detection AI")
    gr.Markdown("Upload one or more chest X-ray images for a patient to classify them as **Normal** or **Pneumonia**.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Patient Information")
            patient_name = gr.Textbox(label="Patient Name", placeholder="e.g., John Doe")
            patient_age = gr.Number(label="Patient Age", minimum=0, maximum=120, step=1)

            gr.Markdown("### 2. Upload Images")
            # Using type="filepath" is simpler and avoids memory issues with large images
            image_input = gr.File(
                label="Upload up to 3 X-Rays",
                file_count="multiple",
                file_types=["image"],
                type="filepath" # Gradio will save uploads to a temp dir and give us the path
            )
            
            submit_btn = gr.Button("Analyze Images", variant="primary")

            if SAMPLE_IMAGES:
                gr.Examples(
                    examples=SAMPLE_IMAGES,
                    inputs=image_input,
                    label="Sample Images (Click one, then click Analyze)",
                    examples_per_page=5
                )

        with gr.Column(scale=1):
            gr.Markdown("### 3. Analysis Results")
            output_label = gr.Label(label="Prediction", num_top_classes=2)
            gr.Markdown("---")
            with gr.Accordion("View Patient Record History", open=False):
                records_html = gr.HTML("Loading records...")
                demo.load(get_records_html, None, records_html) # Load records when the app starts
                refresh_btn = gr.Button("Refresh History")


    # --- Link Components to the Function ---
    submit_btn.click(
        fn=classify_images,
        inputs=[patient_name, patient_age, image_input],
        outputs=[output_label]
    )
    
    # When the refresh button is clicked, re-run the get_records_html function
    refresh_btn.click(fn=get_records_html, inputs=None, outputs=records_html)

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()