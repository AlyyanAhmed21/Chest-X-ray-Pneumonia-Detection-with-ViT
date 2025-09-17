# app.py (Final Version with Syntax Fix)

import gradio as gr
from pathlib import Path
import asyncio
from PIL import Image

# Import backend components
from app.prediction import PredictionPipeline
from app.database import add_patient_record, get_all_records

# --- Initialization ---
prediction_pipeline = PredictionPipeline()
SAMPLE_IMAGE_DIR = Path("sample_images")
try:
    SAMPLE_IMAGES = [str(p) for p in sorted(list(SAMPLE_IMAGE_DIR.glob('*/*.jpeg')))]
    if not SAMPLE_IMAGES:
        raise FileNotFoundError
except FileNotFoundError:
    print("Warning: 'sample_images' directory not found or is empty. Please check setup.sh.")
    SAMPLE_IMAGES = []

# --- Core Logic Functions ---
async def process_analysis(patient_name, patient_age, image_list, is_sample=False):
    if not is_sample and (not patient_name or patient_age is None):
        raise gr.Error("Patient Name and Age are required.")
    if not image_list:
        raise gr.Error("At least one image is required.")
    
    result = prediction_pipeline.predict(image_list)
    if "error" in result:
        raise gr.Error(result["error"])

    final_pred = result["final_prediction"]
    final_conf = result["final_confidence"]
    
    if not is_sample:
        await add_patient_record(str(patient_name), int(patient_age), final_pred, final_conf)

    confidences = {"NORMAL": 0.0, "PNEUMONIA": 0.0}
    confidences[final_pred] = final_conf
    confidences["NORMAL" if final_pred == "PNEUMONIA" else "PNEUMONIA"] = 1 - final_conf
    
    return [
        gr.update(visible=False), # uploader_column
        gr.update(visible=True),  # results_column
        gr.update(value=result["watermarked_images"]), # result_images
        gr.update(value=confidences) # result_label
    ]

async def refresh_history_table():
    records = await get_all_records()
    data = [[r.get('name'), r.get('age'), r.get('prediction_result'), f"{r.get('confidence_score', 0):.2%}", r.get('timestamp').strftime('%Y-%m-%d %H:%M')] for r in records] if records else []
    return gr.update(value=data)

# --- Gradio UI Definition ---
css = """
/* --- Professional Dark Theme & Fonts --- */
:root { --primary-hue: 220 !important; --secondary-hue: 210 !important; --neutral-hue: 210 !important; --body-background-fill: #111827 !important; --block-background-fill: #1F2937 !important; --block-border-width: 1px !important; --border-color-accent: #374151 !important; --background-fill-secondary: #1F2937 !important;}
/* --- Header & Title Styling --- */
#app_header { text-align: center; }
#app_title { font-size: 2.8rem !important; font-weight: 700 !important; color: #FFFFFF !important; padding-top: 1rem; }
#app_subtitle { font-size: 1.2rem !important; color: #9CA3AF !important; margin-bottom: 2rem; }
/* --- Layout, Spacing, and Component Styling --- */
#main_container { gap: 2rem; }
#results_gallery .gallery-item { padding: 0.25rem !important; background-color: #374151; border: 1px solid #374151 !important; }
#bottom_controls { max-width: 600px; margin: 2.5rem auto 1rem auto; }
#bottom_controls .gr-accordion > .gr-block-label { text-align: center !important; display: block !important; }
#sample_gallery .gallery-item { box-shadow: 0 0 5px rgba(0,0,0,0.5); border-radius: 8px !important; }
#sample_gallery .gallery-item.selected { border: 4px solid var(--primary-500) !important; }
"""

# --- THIS IS THE FIX ---
with gr.Blocks(
    theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue"), 
    css=css, 
    title="Pneumonia Detection AI"
) as demo:
    
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
                    ### MLOps-Powered Pneumonia Detection
                    (Your professional description here)
                    ---
                    **Project Team:** Alyyan Ahmed & Munim Akbar
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
        history_df = gr.DataFrame(headers=["Name", "Age", "Prediction", "Confidence", "Date"], row_count=10, interactive=False)

    with gr.Column(visible=False) as samples_page:
        gr.Markdown("# 🖼️ Sample Image Library", elem_classes="app_title")
        gr.Markdown("Select up to 3 images by clicking on them, then click 'Analyze'.")
        sample_gallery = gr.Gallery(
            value=SAMPLE_IMAGES if SAMPLE_IMAGES else [],
            label="Sample Images",
            columns=5, height=400,
            elem_id="sample_gallery"
        )
        selected_samples_textbox = gr.Textbox(visible=False, elem_id="selected_samples_textbox")
        with gr.Row():
            analyze_samples_btn = gr.Button("Analyze Selected Samples", variant="primary")
            back_to_main_btn_samp = gr.Button("⬅️ Back to Main App")
    
    # --- Event Handling Logic ---
    
    select_js = """
    (evt) => {
        const gallery = document.querySelector('#sample_gallery .grid-container');
        const clicked_img = gallery.children[evt.index];
        const selected_paths_input = document.querySelector('#selected_samples_textbox textarea');
        let selected_paths = selected_paths_input.value ? selected_paths_input.value.split(',') : [];
        const current_path = clicked_img.querySelector('img').alt;
        if (clicked_img.classList.contains('selected')) {
            clicked_img.classList.remove('selected');
            selected_paths = selected_paths.filter(p => p !== current_path);
        } else {
            if (selected_paths.length < 3) {
                clicked_img.classList.add('selected');
                selected_paths.push(current_path);
            } else {
                alert("You can select a maximum of 3 images.");
            }
        }
        return selected_paths.join(',');
    }
    """
    sample_gallery.select(fn=None, js=select_js, outputs=[selected_samples_textbox])
    
    def show_patient_info(files): return gr.update(visible=True) if files else gr.update(visible=False)
    image_input.upload(fn=show_patient_info, inputs=image_input, outputs=patient_info_modal)

    async def submit_and_hide_modal(name, age, files):
        analysis_results = await process_analysis(name, age, files)
        return [*analysis_results, gr.update(visible=False)]
    submit_analysis_btn.click(fn=submit_and_hide_modal, inputs=[patient_name_modal, patient_age_modal, image_input], outputs=[uploader_column, results_column, result_images, result_label, patient_info_modal])
    
    cancel_btn.click(lambda: (gr.update(visible=False), None), None, [patient_info_modal, image_input])
    start_over_btn.click(fn=None, js="() => { window.location.reload(); }")

    async def handle_sample_analysis(selected_paths_str: str):
        selected_images = [path for path in selected_paths_str.split(',') if path]
        if not selected_images: raise gr.Error("Please select at least one sample image.")
        if len(selected_images) > 3: raise gr.Error("Please select no more than 3 sample images.")
        
        analysis_results = await process_analysis("Sample User", 0, selected_images, is_sample=True)
        return [gr.update(visible=True), gr.update(visible=False), *analysis_results]
    analyze_samples_btn.click(fn=handle_sample_analysis, inputs=[selected_samples_textbox], outputs=[main_app, samples_page, uploader_column, results_column, result_images, result_label])

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
