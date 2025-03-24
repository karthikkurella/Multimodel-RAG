import gradio as gr
from huggingface_hub import InferenceClient
import time
import requests
from PIL import Image
import io

# Initialize the Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key="***",  # Replace with your actual API key
)

def process_ticket(text, image=None):
    try:
        if image is not None:
            # Resize and compress the image
            img = Image.open(image)
            
            # Resize to smaller dimensions while maintaining aspect ratio
            max_size = 800  # You can adjust this value
            ratio = min(max_size/img.width, max_size/img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            
            # Convert to bytes with compression
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='png', quality=85)  # Lower quality for smaller size
            img_byte_arr = img_byte_arr.getvalue()

            # Properly encode the image as base64
            import base64
            base64_img = base64.b64encode(img_byte_arr).decode('utf-8')

            # Prepare the message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                        }
                    ]
                }
            ]
        else:
            # Text-only message
            messages = [
                {
                    "role": "user",
                    "content": text
                }
            ]

        # Make the API call with retry logic for 503 errors
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                completion = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct",  
                    messages=messages
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    print(f"503 error encountered. Retrying in 30 seconds...")
                    time.sleep(30)
                    retries += 1
                else:
                    raise
        
        if retries == max_retries:
            return "Failed to process request after retries."
        
        # Extract and return the response
        response = completion.choices[0].message
        return response.content

    except Exception as e:
        print(f"Error processing ticket: {e}")
        return f"An error occurred while processing your request: {str(e)}"

def create_interface():
    text_input = gr.Textbox(
        label="Describe your issue",
        placeholder="Describe the problem you're experiencing",
        lines=4,
    )
    
    image_input = gr.Image(label="Upload an image (Optional)", type="filepath")
    
    output = gr.Textbox(label="Suggested Solution", lines=5)
    
    interface = gr.Interface(
        fn=process_ticket,
        inputs=[text_input, image_input],  
        outputs=output,
        title="Multimodal AI Assistant",
        description="Submit a description of your issue and optionally an image to get AI-powered suggestions.",
    )
    
    interface.launch(debug=True)

create_interface()
