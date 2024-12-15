"""
====================================================================
This script serves a fine-tuned LLaMA model as a text generation service using BentoML.

In this version, we've simplified the interface by removing additional sampling parameters 
like temperature, top_k, and top_p. This makes the service easier to integrate into a pipeline, 
relying on default model generation settings while still allowing for a customizable max_length.

Steps:
1. Loads a previously fine-tuned LLaMA model and tokenizer.
2. Defines a single text generation endpoint using BentoML.
3. Supports specifying a max_length for the generated text, but otherwise uses default generation parameters.

Authors: alex.mozerski, daniel.ribeirocabral, victor.rominger, killian.ruffieux, ruben.terceiro
Date: 16.12.2024
====================================================================
"""

import bentoml
from bentoml.io import JSON
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
MODEL_DIR = "fine_tuned_lora_llama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print("Model and tokenizer loaded successfully.")

# Text generation function
def generate_text(prompt, max_length=50, temperature=0.9, top_k=50, top_p=0.9):
    """
    Generates text based on the input prompt using the fine-tuned model.
    
    Parameters:
    - prompt (str): The input text prompt for the model.
    - max_length (int): Maximum length of the generated text (default is 50).
    - temperature (float): Sampling temperature for controlling randomness (default is 0.9).
    - top_k (int): Top-k sampling parameter for selecting likely tokens (default is 50).
    - top_p (float): Nucleus sampling parameter for selecting cumulative probabilities (default is 0.9).
    
    Returns:
    - str: Generated text as a string.
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

# BentoML Service
svc = bentoml.Service("text_generator")

@svc.api(input=JSON(), output=JSON())
def generate(data: dict) -> dict:
    """
    API endpoint to generate text based on input data.
    
    Parameters:
        - data (dict): Input JSON containing the following keys:
            - "prompt" (str): The text prompt for the model.
            - "max_length" (int, optional): Maximum length for generated text (default is 50).
            - "temperature" (float, optional): Sampling temperature (default is 0.9).
            - "top_k" (int, optional): Top-k sampling parameter (default is 50).
            - "top_p" (float, optional): Nucleus sampling parameter (default is 0.9).
        
    Returns:
        - dict: A dictionary containing either the generated text or an error message.
    """
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)
    temperature = data.get("temperature", 0.9)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.9)
    
    if not prompt:
        return {"error": "Prompt is required"}
    
    generated_text = generate_text(prompt, max_length, temperature, top_k, top_p)
    return {"generated_text": generated_text}
