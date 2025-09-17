# app.py (The Definitive Final Version)

import gradio as gr
from pathlib import Path
import asyncio

# Import backend components
from app.prediction import PredictionPipeline
from app.database import add_patient_record, get_all_records, clear_all_records # Import the new function

# --- Initialization ---
prediction_pipeline = PredictionPipeline()
SAMPLE_IMAGE_DIR = Path("sample_images")
try:
    if SAMPLE_IMAGE_DIR.is_dir():
        NORMAL_SAMPLES = [str(p) for p in sorted(list((SAMPLE_IMAGE_DIR / 'NORMAL').glob('*.jpeg')))]
        PNEUMONIA_SAMPLES = [str(p) for p in sorted(list((SAMPLE_IMAGE_DIR / 'PNEUMONIA').glob('*.jpeg')))]
    else: raise FileNotFoundError
except FileNotFoundError:
    print("Warning: 'sample_images' directory not found."); NORMAL_SAMPLES, PNEUMONIA_SAMPLES = [], []

# --- Core Logic Functions (Unchanged) ---
async def process_analysis(patient_name, patient_age, image_list):
    if not patient_name or patient_age is None: raise gr.Error("Patient Name and Age are required.")
    if not image_list: raise gr.Error("At least one image is required.")
    result = prediction_pipeline.predict(image_list)
    if "error" in result: raise gr.Error(result.get("details", result["error"]))
    final_pred, final_conf = result["final_prediction"], result["final_confidence"]
    await add_patient_record(str(patient_name), int(patient_age), final_pred, final_conf)
    confidences = {"NORMAL": 0.0, "PNEUMONIA": 0.0}; confidences[final_pred] = final_conf; confidences["NORMAL" if final_pred == "PNEUMONIA" else "PNEUMONIA"] = 1 - final_conf
    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=result["watermarked_images"]), gr.update(value=confidences)]
async def refresh_history_table():
    records = await get_all_records()
    data = [[r.get('name'), r.get('age'), r.get('prediction_result'), f"{r.get('confidence_score', 0):.2%}", r.get('timestamp').strftime('%Y-%m-%d %H:%M')] for r in records] if records else []
    return gr.update(value=data)

# --- Gradio UI Definition ---
css = """
/* --- Professional Dark Theme & Fonts --- */
:root { --primary-hue: 220 !important; --secondary-hue: 210 !important; --neutral-hue: 210 !important; --body-background-fill: #111827 !important; --block-background-fill: #1F2337 !important; --block-border-width: 1px !important; --border-color-accent: #374151 !important; --background-fill-secondary: #1F2937 !important;}
/* --- Header & Title Styling (THE FIX) --- */
#app_header { text-align: center; max-width: 900px; margin: 0 auto; }
#app_title { font-size: 3rem !important; font-weight: 800 !important; color: #FFFFFF !important; padding-top: 1rem; }
#app_subtitle { font-size: 1.25rem !important; color: #9CA3AF !important; margin-bottom: 2rem; }
/* --- Layout and Spacing --- */
#main_container { gap: 2rem; max-width: 700px; margin: 0 auto; } /* Made it slightly narrower */
#results_gallery .gallery-item { padding: 0.25rem !important; background-color: #374151; border: 1px solid #374151 !important; }
#bottom_controls { max-width: 500px; margin: 2.5rem auto 1rem auto; }
"""
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue"), css=css, title="Pneumonia Detection AI") as demo:
    
    with gr.Column() as main_app:
        with gr.Column(elem_id="app_header"):
            gr.Markdown("# 🩺 Pneumonia Detection AI", elem_id="app_title")
            gr.Markdown("An AI-powered tool to assist in the diagnosis of pneumonia.", elem_id="app_subtitle")
        with gr.Row(elem_id="main_container"):
            with gr.Column(scale=1) as uploader_column:
                gr.Markdown("### Upload Patient X-Rays")
                image_input = gr.File(label="Upload up to 3 Images", file_count="multiple", file_types=["image"], type="filepath")
            with gr.Column(scale=2, visible=False) as results_column:
                gr.Markdown("### Analysis Results")
                result_images = gr.Gallery(label="Analyzed Images", columns=3, object_fit="contain", height=350, elem_id="results_gallery")
                result_label = gr.Label(label="Overall Prediction", num_top_classes=2)
                start_over_btn = gr.Button("Start New Analysis", variant="secondary")
        with gr.Group(visible=False) as patient_info_modal:
            gr.Markdown("## Enter Patient Details", elem_classes="text-center")
            patient_name_modal = gr.Textbox(label="Patient Name", placeholder="e.g., John Doe")
            patient_age_modal = gr.Number(label="Patient Age", minimum=0, maximum=120, step=1)
            with gr.Row():
                submit_analysis_btn = gr.Button("Analyze Images", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")
        with gr.Column(elem_id="bottom_controls"):
            with gr.Accordion("About this Tool", open=False):
                gr.Markdown(
                    """
                    ### 🩺 MLOps-Powered Pneumonia Detection  
                    
                    This project showcases a complete **end-to-end MLOps pipeline** for medical image classification.  
                    It leverages a cutting-edge **Vision Transformer (ViT)** model, fine-tuned on publicly available chest X-ray datasets, to classify images into **Normal** or **Pneumonia** cases.  
                    
                    ⚠️ **Disclaimer:** This application is intended **solely for educational and demonstration purposes**. It is **not a medical diagnostic tool** and must not be used as a substitute for professional medical advice.  
                    
                    ---
                    
                    ### 👥 Project Team  
                    - **Alyyan Ahmed** — ML Engineer & Developer  
                    - **Munim Akbar** — ML Engineer & Developer  
                    """
                )
            with gr.Row():
                samples_btn = gr.Button("Try Sample Images")
                history_btn = gr.Button("View Patient History")

    with gr.Column(visible=False) as history_page:
        gr.Markdown("# 📜 Patient Record History", elem_classes="app_title")
        with gr.Row():
            back_to_main_btn_hist = gr.Button("⬅️ Back to Main App")
            refresh_history_btn = gr.Button("Refresh History")
            # --- NEW: Clear History Button ---
            clear_history_btn = gr.Button("⚠️ Clear All History", variant="stop")
        history_df = gr.DataFrame(headers=["Name", "Age", "Prediction", "Confidence", "Date"], row_count=10, interactive=False)

    with gr.Column(visible=False) as samples_page:
        gr.Markdown("# 🖼️ Sample Image Library", elem_classes="app_title")
        gr.Markdown("You can download these sample images to test the tool on the main page.")
        back_to_main_btn_samp = gr.Button("⬅️ Back to Main App")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Normal Cases")
                for img_path in NORMAL_SAMPLES:
                    gr.File(value=img_path, label=Path(img_path).name, interactive=False)
            with gr.Column():
                gr.Markdown("### Pneumonia Cases")
                for img_path in PNEUMONIA_SAMPLES:
                    gr.File(value=img_path, label=Path(img_path).name, interactive=False)
    
    # --- Event Handling Logic ---
    
    # ... (main page handlers are correct)
    def show_patient_info(files): return gr.update(visible=True) if files else gr.update(visible=False)
    image_input.upload(fn=show_patient_info, inputs=image_input, outputs=patient_info_modal)
    async def submit_and_hide_modal(name, age, files):
        analysis_results = await process_analysis(name, age, files); return [*analysis_results, gr.update(visible=False)]
    submit_analysis_btn.click(fn=submit_and_hide_modal, inputs=[patient_name_modal, patient_age_modal, image_input], outputs=[uploader_column, results_column, result_images, result_label, patient_info_modal])
    cancel_btn.click(lambda: (gr.update(visible=False), None), None, [patient_info_modal, image_input])
    start_over_btn.click(fn=None, js="() => { window.location.reload(); }")
    
    # --- Page Navigation (correct) ---
    all_pages = [main_app, history_page, samples_page]
    async def show_history_page_and_refresh():
        records_update = await refresh_history_table(); return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), records_update]
    def show_samples_page(): return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
    def show_main_page(): return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]
    
    history_btn.click(fn=show_history_page_and_refresh, outputs=all_pages + [history_df])
    samples_btn.click(fn=show_samples_page, outputs=all_pages)
    back_to_main_btn_hist.click(fn=show_main_page, outputs=all_pages)
    back_to_main_btn_samp.click(fn=show_main_page, outputs=all_pages)
    
    # --- History Page Logic ---
    refresh_history_btn.click(fn=refresh_history_table, outputs=history_df)
    
    # --- NEW: Clear History Logic ---
    async def clear_history_and_refresh():
        deleted_count = await clear_all_records()
        gr.Info(f"Successfully deleted {deleted_count} records.")
        # After deleting, immediately refresh the table to show it's empty
        return await refresh_history_table()
    clear_history_btn.click(fn=clear_history_and_refresh, outputs=history_df)
    
    demo.load(fn=refresh_history_table, outputs=history_df)

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()
