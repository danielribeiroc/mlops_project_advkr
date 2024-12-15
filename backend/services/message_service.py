from models.message import Message
from datetime import datetime
from services.rag_functions import init_retriever, load_files_to_dict, rag_request
import requests
import bentoml
from bentoml.io import JSON
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
MODEL_DIR = "fine_tuned_lora_llama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_message(message: Message):
    print(message.prompt)
    if "hello" in message.prompt.lower():
        return {"sender": "bot", "generated_text": "Hello! How can I assist you today?", "timestamp": datetime.utcnow()}
    return {"sender": "bot", "generated_text": "Hello! How can I assist you today?", "timestamp": datetime.utcnow()}

    #return rag_call(message, pipeline_rag)

def rag_call(message: Message, pipeline_rag):
    print("Loading files...")
    #documents = load_files_to_dict("s3://2024-mlops-projet-advkr/models--meta-llama--Llama-3.2-1B", source="s3")
    documents = load_files_to_dict("./datasets", source="local") # TODO
    print("Files loaded.")
    print("Initializing retriever...")
    retriever = init_retriever(documents)
    print("Retriever initialized.")
    print("Sending RAG request...")
    answer = rag_request(message.text, retriever, pipeline_rag)
    print("RAG request sent.")

    return {"sender": "rag", "text": answer, "timestamp": datetime.utcnow()}

def lora_call(message: Message):
    """
    Handles text generation using the LoRA fine-tuned model on GPU.
    """
    print("Loading model and tokenizer...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)  # Move to GPU
    model.eval()  # Set the model to evaluation mode (important for generation)

    print("Model and tokenizer loaded successfully.")

    # Extract parameters from the Message object
    prompt = message.prompt
    max_length = message.max_length if message.max_length else 50  # Set default if None
    temperature = message.temperature if message.temperature else 0.4  # Set default if None

    # Generate text
    generated_text = generate_text(prompt, tokenizer, model, max_length, temperature)

    print("Generated text:", generated_text)
    return {"sender": "bot", "generated_text": generated_text, "timestamp": datetime.utcnow()}


# Updated generate_text function
def generate_text(prompt, tokenizer, model, max_length=50, temperature=0.9, top_k=50, top_p=0.9):
    """
    Generates text from a given prompt using the fine-tuned model.
    """
    with torch.no_grad():  # No gradient calculation for inference
        # Tokenize input text and move tensors to GPU
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}  # Move to GPU

        # Generate text
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            top_k=top_k,
            top_p=top_p,
        )
    # Decode the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

