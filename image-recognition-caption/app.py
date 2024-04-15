import gradio as gr
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor, ViTForImageClassification
import torch

# Initialize device and models for captioning
device = 'cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTImageProcessor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
caption_model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

# Initialize the image recognition model
recognition_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)

def get_caption(image):
    # Generate a caption from the image
    image = image.convert('RGB')
    image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    caption_ids = caption_model.generate(image_tensor, max_length=128, num_beams=3)[0]
    caption_text = tokenizer.decode(caption_ids, skip_special_tokens=True)
    return caption_text

def classify_image(image):
    # Prepare the image for classification
    image = image.convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = recognition_model(**inputs.to(device))
    
    # Get top 5 results
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_probs, top_labels = probs.topk(5)
    
    # Convert to readable labels and probabilities
    results = [(recognition_model.config.id2label[label.item()], prob.item()) for label, prob in zip(top_labels[0], top_probs[0])]
    return dict(results)

# Set up Gradio interface
title = "Image Captioning and Recognition"
with gr.Blocks(title=title) as demo:
    with gr.Row():
        gr.Markdown("# Simple Image Caption & Image Recognition App")
    with gr.Row():
        gr.Markdown("### This app allows you to upload an image and see it's caption and classification.")
    with gr.Column():
        image_input = gr.Image(label="Upload any Image", type='pil')
        get_caption_btn = gr.Button("Get Caption")
        caption_output = gr.Textbox(label="Caption")
        classify_btn = gr.Button("Classify Image")
        classification_output = gr.Label(label="Predicted Labels and Probabilities")
        
        get_caption_btn.click(get_caption, inputs=image_input, outputs=caption_output)
        classify_btn.click(classify_image, inputs=image_input, outputs=classification_output)

demo.launch()