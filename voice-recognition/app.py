import gradio as gr
import logging
from transformers import pipeline
import torch

asr = pipeline(task="automatic-speech-recognition",
               model="openai/whisper-tiny.en")


# Adjusted function assuming 'asr' expects a file path as input
def transcribe_speech(audio_file_path):
    if not audio_file_path:
        logging.error("No audio file provided.")
        return "No audio found, please retry."
    try:
        logging.info(f"Processing file: {audio_file_path}")
        output = asr(audio_file_path)  # Assuming `asr` directly takes a file path
        return output["text"]
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return f"Error processing the audio file: {str(e)}"

logging.basicConfig(level=logging.INFO)

with gr.Blocks() as demo:
    with gr.Row():
        mic = gr.Audio(label="Record from Microphone or Upload File", type="filepath")
        transcribe_button = gr.Button("Transcribe Audio")

    transcription = gr.Textbox(label="Transcription", lines=3, placeholder="Transcription will appear here...")

    transcribe_button.click(transcribe_speech, inputs=mic, outputs=transcription)

demo.launch(share=True)
