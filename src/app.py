import gradio as gr
from huggingface_hub import InferenceClient
import time
import requests
from PIL import Image
import io
import base64
import os
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import json

# Initialize the Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key="*****",  # Replace with your actual API key
)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a simple vector database using FAISS
dimension = 384  # Dimension of the embeddings from all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
pdf_chunks = []  # Store text chunks
pdf_metadata = []  # Store metadata about the PDFs

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Find the last period or newline to make cleaner chunks
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            split_point = max(last_period, last_newline)
            
            if split_point > start:
                end = split_point + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length
    
    return chunks

def store_pdf(pdf_path, pdf_name):
    """Process PDF and store in vector database"""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    # Create embeddings for each chunk
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode([chunk])[0]
        
        # Add to FAISS index
        index.add(np.array([embedding], dtype=np.float32))
        
        # Store the text and metadata
        pdf_chunks.append(chunk)
        pdf_metadata.append({
            "pdf_name": pdf_name,
            "chunk_id": i,
            "total_chunks": len(chunks)
        })
    
    return f"Processed and stored '{pdf_name}' with {len(chunks)} chunks"

def retrieve_context(query, top_k=3):
    """Retrieve relevant context based on the query"""
    if index.ntotal == 0:
        return "No documents have been uploaded yet."
    
    # Encode the query
    query_embedding = embedding_model.encode([query])[0]
    
    # Search for similar chunks
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    
    # Retrieve the chunks
    context = ""
    for idx in indices[0]:
        if idx < len(pdf_chunks):
            context += pdf_chunks[idx] + "\n\n"
    
    return context

def process_ticket(text, image=None, pdf=None):
    try:
        # Handle PDF upload
        pdf_result = ""
        if pdf is not None:
            pdf_name = os.path.basename(pdf)
            pdf_result = store_pdf(pdf, pdf_name)
        
        # Process image if provided
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
            # For text-only or RAG-based queries
            if "?" in text and index.ntotal > 0:
                # This appears to be a question, use RAG
                context = retrieve_context(text)
                
                # Create a RAG prompt
                rag_prompt = f"""Answer the following question based on the provided context:
                
Question: {text}

Context:
{context}

Please provide a clear and concise answer based only on the information in the context. If the context doesn't contain relevant information, say so.
                """
                
                messages = [
                    {
                        "role": "user",
                        "content": rag_prompt
                    }
                ]
            else:
                # Regular text-only message
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
        response = completion.choices[0].message.content
        
        # Add PDF processing result if applicable
        if pdf_result:
            response = f"{pdf_result}\n\n{response}"
            
        return response

    except Exception as e:
        print(f"Error processing ticket: {e}")
        return f"An error occurred while processing your request: {str(e)}"

def create_interface():
    with gr.Blocks(title="Multimodal AI Assistant with PDF RAG") as demo:
        gr.Markdown("# Multimodal AI Assistant with PDF RAG")
        gr.Markdown("Submit a description, upload an image, or ask questions about your PDFs.")
        
        with gr.Tab("Chat & Upload"):
            text_input = gr.Textbox(
                label="Message or Question",
                placeholder="Describe an issue or ask a question about your uploaded PDFs",
                lines=4,
            )
            
            with gr.Row():
                image_input = gr.Image(label="Upload an image (Optional)", type="filepath")
                pdf_input = gr.File(label="Upload a PDF (Optional)", file_types=[".pdf"])
            
            submit_btn = gr.Button("Submit")
            output = gr.Textbox(label="Response", lines=10)
            
            submit_btn.click(
                fn=process_ticket,
                inputs=[text_input, image_input, pdf_input],
                outputs=output
            )
        
        with gr.Tab("PDF Database"):
            gr.Markdown("### Uploaded PDFs")
            
            def list_pdfs():
                if len(pdf_metadata) == 0:
                    return "No PDFs uploaded yet."
                
                # Get unique PDF names
                unique_pdfs = set(item["pdf_name"] for item in pdf_metadata)
                return "\n".join(f"- {pdf}" for pdf in unique_pdfs)
            
            pdf_list = gr.Textbox(label="Uploaded PDFs", lines=5)
            refresh_btn = gr.Button("Refresh List")
            refresh_btn.click(fn=list_pdfs, inputs=[], outputs=pdf_list)
    
    demo.launch(debug=True)

create_interface()
