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

# Directory containing the fine-tuned model and tokenizer
MODEL_DIR = "fine_tuned_lora_llama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and tokenizer...")
# Load the tokenizer, which encodes text into model-understandable IDs
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load the fine-tuned model and move it to the appropriate device
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print("Model and tokenizer loaded successfully.")

def generate_text(prompt, max_length=50):
    """
    Generate text using the fine-tuned model with default generation parameters.

    Parameters:
        prompt (str): The initial prompt text to guide the model's generation.
        max_length (int): The maximum length of the output sequence, including prompt tokens.

    Returns:
        str: The generated text.
    """
    with torch.no_grad():
        # Encode the prompt into token IDs
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        # Generate output text without extra sampling parameters for simplicity
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define a BentoML Service
svc = bentoml.Service("text_generator")

@svc.api(input=JSON(), output=JSON())
def generate(data: dict) -> dict:
    """
    A BentoML endpoint that generates text from a given prompt and optional max_length.

    Example input:
    {
        "prompt": "Once upon a time,",
        "max_length": 100
    }

    Output:
    {
        "generated_text": "..."
    }

    Parameters:
        data (dict): The JSON payload containing:
            - "prompt" (str, required): The starting text for generation.
            - "max_length" (int, optional): Maximum length of the generated text. Defaults to 50.

    Returns:
        dict: A JSON response with the generated text or an error if no prompt is provided.
    """
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)

    if not prompt:
        return {"error": "Prompt is required"}

    generated_text = generate_text(prompt, max_length)
    return {"generated_text": generated_text}
