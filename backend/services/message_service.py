from models.message import Message
from datetime import datetime
from services.rag_functions import init_retriever, load_files_to_dict, rag_request
import requests
import bentoml
from bentoml.io import JSON
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = "fine_tuned_lora_llama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_message(message: Message):
    if "hello" in message.prompt.lower():
        return {"sender": "bot", "generated_text": "Hello! How can I assist you today?", "timestamp": datetime.utcnow()}
    return {"sender": "bot", "generated_text": "Hello! How can I assist you today?", "timestamp": datetime.utcnow()}

def rag_call(message: Message, pipeline_rag):
    documents = load_files_to_dict("./datasets", source="local")
    retriever = init_retriever(documents)
    answer = rag_request(message.text, retriever, pipeline_rag)
    return {"sender": "rag", "text": answer, "timestamp": datetime.utcnow()}

def lora_call(message: Message):
    """
    Handles text generation using the LoRA fine-tuned model on GPU.
    """
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    print("Model and tokenizer loaded successfully.")
    prompt = message.prompt
    max_length = message.max_length if message.max_length else 50
    temperature = message.temperature if message.temperature else 0.4
    generated_text = generate_text(prompt, tokenizer, model, max_length, temperature)

    print("Generated text:", generated_text)
    return {"sender": "bot", "generated_text": generated_text, "timestamp": datetime.utcnow()}

def generate_text(prompt, tokenizer, model, max_length=50, temperature=0.9, top_k=50, top_p=0.9):
    """
    Generates text from a given prompt using the fine-tuned model.
    """
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

