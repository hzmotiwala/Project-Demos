import gradio as gr

def transcribe_speech(audio_file):
    if audio_file is None:
        return "No audio found, please retry."
    output = asr(audio_file.name)  # Assuming `asr` is your speech recognition function
    return output["text"]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            mic = gr.Audio(source="microphone", type="filepath", label="Record from Microphone")
            mic_button = gr.Button("Transcribe from Microphone")
        with gr.Column():
            upload = gr.Audio(source="upload", type="filepath", label="Upload Audio File")
            upload_button = gr.Button("Transcribe Uploaded File")

    transcription = gr.Textbox(label="Transcription", lines=3, placeholder="Transcription will appear here...")

    mic_button.click(transcribe_speech, inputs=mic, outputs=transcription)
    upload_button.click(transcribe_speech, inputs=upload, outputs=transcription)

demo.launch(share=True)
