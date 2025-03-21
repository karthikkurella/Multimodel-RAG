import torch
from PIL import Image
import gradio as gr
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login

login(token = "hf_vyFCfVXSSIJmnjigUduytmhuLbqFLdSCg")

def load_model():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"  
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if GPU is available

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
        device_map="auto",  # Automatically map to available device
        offload_folder="offload",  # Offload to disk if necessary
    )
    
    model.tie_weights()  # Tying weights for efficiency
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Model loaded on: {device}")
    
    return model, processor

def process_ticket(text, image=None):
    model, processor = load_model()
    
    try:
        if image:
            # Resize the image for consistency
            image = image.convert("RGB").resize((224, 224))
            prompt = f"<|image|><|begin_of_text|>{text}"
            # Process both the image and text input
            inputs = processor(images=[image], text=prompt, return_tensors="pt").to(model.device)
        else:
            prompt = f"<|begin_of_text|>{text}"
            # Process text-only input
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        
        # Generate response (restrict token length for faster output)
        outputs = model.generate(**inputs, max_new_tokens=200)
        # Decode the response from tokens to text
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        print(f"Error processing ticket: {e}")
        return "An error occurred while processing your request."

def create_interface():
    text_input = gr.Textbox(
        label="Describe your issue",
        placeholder="Describe the problem you're experiencing",
        lines=4,
    )
    
    image_input = gr.Image(label="Upload a Screenshot (Optional)", type="pil")
    
    # Output element
    output = gr.Textbox(label="Suggested Solution", lines=5)
    
    # Create the Gradio interface
    interface = gr.Interface(
        fn=process_ticket,  # Function to process inputs
        inputs=[text_input, image_input],  # User inputs (text and image)
        outputs=output,  # AI-generated output
        title="Multimodal AI Assistant",
        description="Submit a description of your issue, along with an optional screenshot, and get AI-powered suggestions.",
    )
    
    # Launch the interface with debug mode
    interface.launch(debug=True)

create_interface()