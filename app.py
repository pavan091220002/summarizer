import gradio as gr
from utils import process_company
import json

def run_analysis(company: str):
    """Run analysis and return formatted output."""
    result = process_company(company)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    audio_file = result["Audio"]
    return report, audio_file

# Gradio Interface
with gr.Blocks(title="News Summarizer") as demo:
    gr.Markdown("# News Summarization & TTS Application")
    company_input = gr.Textbox(label="Enter Company Name", placeholder="e.g., Tesla")
    submit_btn = gr.Button("Analyze")
    
    report_output = gr.Textbox(label="Sentiment Report", lines=20)
    audio_output = gr.Audio(label="Hindi TTS Output")
    
    submit_btn.click(
        fn=run_analysis,
        inputs=company_input,
        outputs=[report_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch()