# app.py (The Final Polished Version)

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
    if SAMPLE_IMAGE_DIR.is_dir():
        SAMPLE_IMAGES = [str(p) for p in sorted(list(SAMPLE_IMAGE_DIR.glob('*/*.jpeg')))]
    else: raise FileNotFoundError
except FileNotFoundError:
    print("Warning: 'sample_images' directory not found."); SAMPLE_IMAGES = []

# --- Core Logic Functions (Unchanged and Correct) ---
async def process_analysis(patient_name, patient_age, image_list, is_sample=False):
    if not is_sample and (not patient_name or patient_age is None):
        raise gr.Error("Patient Name and Age are required.")
    if not image_list:
        raise gr.Error("At least one image is required.")
    
    result = prediction_pipeline.predict(image_list)
    if "error" in result:
        raise gr.Error(result.get("details", result["error"]))

    final_pred = result["final_prediction"]
    final_conf = result["final_confidence"]
    
    if not is_sample:
        await add_patient_record(str(patient_name), int(patient_age), final_pred, final_conf)

    confidences = {"NORMAL": 0.0, "PNEUMONIA": 0.0}
    confidences[final_pred] = final_conf
    confidences["NORMAL" if final_pred == "PNEUMONIA" else "PNEUMONIA"] = 1 - final_conf
    
    return [
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=result["watermarked_images"]),
        gr.update(value=confidences)
    ]

async def refresh_history_table():
    records = await get_all_records()
    data_for_df = []
    if records:
        data_for_df = [[r.get('name'), r.get('age'), r.get('prediction_result'), f"{r.get('confidence_score', 0):.2%}", r.get('timestamp').strftime('%Y-%m-%d %H:%M')] for r in records]
    return gr.update(value=data_for_df)

# --- Gradio UI Definition ---
css = """
/* --- Professional Dark Theme & Fonts --- */
:root { --primary-hue: 220 !important; --secondary-hue: 210 !important; --neutral-hue: 210 !important; --body-background-fill: #111827 !important; --block-background-fill: #1F2937 !important; --block-border-width: 1px !important; --border-color-accent: #374151 !important; --background-fill-secondary: #1F2937 !important;}
/* --- Header & Title Styling --- */
#app_header { text-align: center; max-width: 900px; margin: 0 auto; } /* --- FIX: Center the header column --- */
#app_title { font-size: 2.8rem !important; font-weight: 700 !important; color: #FFFFFF !important; padding-top: 1rem; }
#app_subtitle { font-size: 1.2rem !important; color: #9CA3AF !important; margin-bottom: 2rem; }
/* --- Layout, Spacing, and Component Styling --- */
#main_container { gap: 2rem; max-width: 900px; margin: 0 auto; } /* Center the main content */
#results_gallery .gallery-item { padding: 0.25rem !important; background-color: #374151; border: 1px solid #374151 !important; }
#bottom_controls { max-width: 500px; margin: 2.5rem auto 1rem auto; }
#sample_gallery .gallery-item { border: 4px solid transparent; transition: border-color 0.3s ease; }
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
                gr.Markdown("...") # Professional description here
            with gr.Row():
                samples_btn = gr.Button("Try Sample Images")
                history_btn = gr.Button("View Patient History")

    with gr.Column(visible=False) as history_page:
        gr.Markdown("# 📜 Patient Record History", elem_classes="app_title")
        with gr.Row():
            back_to_main_btn_hist = gr.Button("⬅️ Back to Main App")
            refresh_history_btn = gr.Button("Refresh History")
        history_df = gr.DataFrame(headers=["Name", "Age", "Prediction", "Confidence", "Date"], row_count=10, interactive=False)

    # --- SAMPLES PAGE (THE DEFINITIVE FIX) ---
    with gr.Column(visible=False) as samples_page:
        gr.Markdown("# 🖼️ Sample Image Library", elem_classes="app_title")
        gr.Markdown("Click an image to run an anonymous analysis.")
        
        # We will use the gallery's native .select() event.
        sample_gallery = gr.Gallery(
            value=SAMPLE_IMAGES,
            label="Sample Images",
            columns=5, height=400,
            allow_preview=True, # Allows the nice popup view
            elem_id="sample_gallery"
        )
        # We add a hidden button that our code will "click"
        hidden_sample_analyze_btn = gr.Button("Analyze Sample", visible=False)
        back_to_main_btn_samp = gr.Button("⬅️ Back to Main App")
    
    # --- Event Handling Logic ---
    
    # ... (upload, modal, start over handlers are correct)
    def show_patient_info(files): return gr.update(visible=True) if files else gr.update(visible=False)
    image_input.upload(fn=show_patient_info, inputs=image_input, outputs=patient_info_modal)
    async def submit_and_hide_modal(name, age, files):
        analysis_results = await process_analysis(name, age, files); return [*analysis_results, gr.update(visible=False)]
    submit_analysis_btn.click(fn=submit_and_hide_modal, inputs=[patient_name_modal, patient_age_modal, image_input], outputs=[uploader_column, results_column, result_images, result_label, patient_info_modal])
    cancel_btn.click(lambda: (gr.update(visible=False), None), None, [patient_info_modal, image_input])
    start_over_btn.click(fn=None, js="() => { window.location.reload(); }")

    # --- SAMPLE PAGE LOGIC (THE DEFINITIVE FIX) ---
    
    # When a sample image is clicked, this function runs.
    # It takes the event data, which contains the path of the clicked image.
    # Its ONLY job is to programmatically "click" the hidden analysis button.
    def on_sample_select(evt: gr.SelectData):
        # We return the path of the selected image. This value will become the input
        # for the hidden_sample_analyze_btn's click event.
        return evt.value
    
    # The .select() event's output is now the INPUT to the hidden button's .click() event.
    sample_gallery.select(
        fn=on_sample_select,
        None,
        hidden_sample_analyze_btn
    )
    
    # The hidden button's click event runs the actual analysis
    async def handle_sample_analysis(selected_image_path: str):
        if not selected_image_path: # This handles the case where nothing is selected
            raise gr.Error("Sample image path is missing.")
            
        analysis_results = await process_analysis("Sample User", 0, [selected_image_path], is_sample=True)
        
        return [
            gr.update(visible=True),   # main_app
            gr.update(visible=False),  # samples_page
            *analysis_results
        ]
        
    hidden_sample_analyze_btn.click(
        fn=handle_sample_analysis, 
        inputs=[hidden_sample_analyze_btn], # The button's value is the path
        outputs=[main_app, samples_page, uploader_column, results_column, result_images, result_label]
    )
    
    # --- Page Navigation (Unchanged and Correct) ---
    all_pages = [main_app, history_page, samples_page]
    async def show_history_page_and_refresh(): records_update = await refresh_history_table(); return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), records_update]
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
