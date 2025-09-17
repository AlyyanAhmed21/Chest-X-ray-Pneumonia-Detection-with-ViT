# app.py (Final Version with Checkbox Samples and Watermark Fix)

import gradio as gr
from pathlib import Path
from huggingface_hub import snapshot_download
import asyncio

from app.prediction import PredictionPipeline
from app.database import add_patient_record, get_all_records

# --- Initialization ---
prediction_pipeline = PredictionPipeline()
HF_DATASET_REPO = "ALYYAN/chest-xray-pneumonia-samples"
try:
    SAMPLE_IMAGE_DIR = Path(snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset"))
    SAMPLE_IMAGES = [str(p) for p in list(SAMPLE_IMAGE_DIR.glob('*/*.jpeg'))]
except Exception as e:
    print(f"Could not download sample images: {e}")
    SAMPLE_IMAGES = []

# --- Core Logic (Async Functions - Unchanged) ---
async def process_analysis(patient_name, patient_age, image_list, is_sample=False):
    # ... (code is the same)
    if not is_sample and (not patient_name or patient_age is None or str(patient_age).strip() == ""): raise gr.Error("Patient Name and Age are required.")
    if not image_list: raise gr.Error("At least one image is required.")
    result = prediction_pipeline.predict(image_list)
    if "error" in result: raise gr.Error(result["error"])
    final_pred, final_conf = result["final_prediction"], result["final_confidence"]
    if not is_sample: await add_patient_record(str(patient_name), int(patient_age), final_pred, final_conf)
    confidences = {"NORMAL": 0.0, "PNEUMONIA": 0.0}; confidences[final_pred] = final_conf; confidences["NORMAL" if final_pred == "PNEUMONIA" else "PNEUMONIA"] = 1 - final_conf
    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=result["watermarked_images"]), gr.update(value=confidences)]
async def refresh_history_table():
    # ... (code is the same)
    records = await get_all_records()
    data = [[r.get('name'), r.get('age'), r.get('prediction_result'), f"{r.get('confidence_score', 0):.2%}", r.get('timestamp').strftime('%Y-%m-%d %H:%M')] for r in records] if records else []
    return gr.update(value=data)

# --- Gradio UI Definition ---
css = "..." # (CSS is the same as the previous correct version)
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue"), css=css, title="Pneumonia Detection AI") as demo:
    
    with gr.Column() as main_app:
        # ... (Main page layout is the same)
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
                gr.Markdown("...") # (Your professional description here)
            with gr.Row():
                samples_btn = gr.Button("Try Sample Images")
                history_btn = gr.Button("View Patient History")
                
    with gr.Column(visible=False) as history_page:
        # ... (History page layout is the same)
        gr.Markdown("# 📜 Patient Record History", elem_classes="app_title")
        with gr.Row():
            back_to_main_btn_hist = gr.Button("⬅️ Back to Main App")
            refresh_history_btn = gr.Button("Refresh History")
        history_df = gr.DataFrame(headers=["Name", "Age", "Prediction", "Confidence", "Date"], row_count=10, interactive=False)

    # --- SAMPLES PAGE (THE FIX) ---
    with gr.Column(visible=False) as samples_page:
        gr.Markdown("# 🖼️ Sample Image Library", elem_classes="app_title")
        gr.Markdown("Select up to 3 images, then click 'Analyze Selected Samples'.")
        
        # Use a CheckboxGroup with images as choices
        sample_checkboxes = gr.CheckboxGroup(
            label="Sample Images",
            choices=[(Image.open(p), p) for p in SAMPLE_IMAGES], # Tuple of (PIL Image for display, path for value)
            type="value"
        )
        
        with gr.Row():
            analyze_samples_btn = gr.Button("Analyze Selected Samples", variant="primary")
            back_to_main_btn_samp = gr.Button("⬅️ Back to Main App")
    
    # --- Event Handling Logic ---
    
    # ... (upload, modal, start_over handlers are correct)
    def show_patient_info(files): return gr.update(visible=True) if files else gr.update(visible=False)
    image_input.upload(fn=show_patient_info, inputs=image_input, outputs=patient_info_modal)
    async def submit_and_hide_modal(name, age, files):
        analysis_results = await process_analysis(name, age, files)
        return [*analysis_results, gr.update(visible=False)]
    submit_analysis_btn.click(fn=submit_and_hide_modal, inputs=[patient_name_modal, patient_age_modal, image_input], outputs=[uploader_column, results_column, result_images, result_label, patient_info_modal])
    cancel_btn.click(lambda: (gr.update(visible=False), None), None, [patient_info_modal, image_input])
    start_over_btn.click(fn=None, js="() => { window.location.reload(); }")

    # --- SAMPLE PAGE LOGIC (THE FIX) ---
    async def handle_sample_analysis(selected_images: list):
        # selected_images is now a list of file paths from the checkbox group
        if not selected_images: raise gr.Error("Please select at least one sample image.")
        if len(selected_images) > 3: raise gr.Error("Please select no more than 3 sample images.")
        
        analysis_results = await process_analysis("Sample User", 0, selected_images, is_sample=True)
        
        return {
            main_app: gr.update(visible=True), 
            samples_page: gr.update(visible=False),
            # Unpack dictionary updates for specific components
            uploader_column: analysis_results[0],
            results_column: analysis_results[1],
            result_images: analysis_results[2],
            result_label: analysis_results[3],
        }
    analyze_samples_btn.click(fn=handle_sample_analysis, inputs=[sample_checkboxes], outputs=[main_app, samples_page, uploader_column, results_column, result_images, result_label])

    # ... (Page Navigation is correct)
    all_pages = [main_app, history_page, samples_page]
    async def show_history_page_and_refresh():
        records_update = await refresh_history_table()
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), records_update]
    def show_samples_page(): return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
    def show_main_page(): return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]
    history_btn.click(fn=show_history_page_and_refresh, outputs=all_pages + [history_df])
    samples_btn.click(fn=show_samples_page, outputs=all_pages)
    back_to_main_btn_hist.click(fn=show_main_page, outputs=all_pages)
    back_to_main_btn_samp.click(fn=show_main_page, outputs=all_pages)
    refresh_history_btn.click(fn=refresh_history_table, outputs=history_df)
    demo.load(fn=refresh_history_table, outputs=history_df)

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()
