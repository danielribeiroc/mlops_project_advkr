"""
Deploying and Saving a Fine-tuned LLaMA Language Model with BentoML
===================================================================

This script demonstrates how to integrate a fine-tuned LLaMA language model 
into a BentoML workflow. It performs the following steps:

1. Loads the fine-tuned LLaMA model and its tokenizer from a local directory.
2. Places the model on a GPU if available, otherwise uses the CPU.
3. Saves the model and tokenizer into the BentoML model store, allowing for 
   streamlined deployment and serving through BentoML services.

Prerequisites:
- Install the following packages: transformers, bentoml, torch

Authors: alex.mozerski, daniel.ribeirocabral, victor.rominger, killian.ruffieux, ruben.terceiro
Date: 16.12.2024

====================================================================
"""

import bentoml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Directory where the fine-tuned model and tokenizer files are stored
MODEL_DIR = "fine_tuned_lora_llama"

# Automatically select the best available device (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and model...")
# Load the tokenizer from our fine-tuned model directory. The tokenizer helps 
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load the model that was previously fine-tuned with LoRA adaptation. 
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)

# Set the model to evaluation mode. This turns off behaviors that are useful during training 
model.eval()

print("Tokenizer and model loaded successfully.")
print("Saving model to BentoML model store...")

# Use BentoML to save the model. BentoML stores the model along with its metadata 
model_ref = bentoml.pytorch.save_model(
    name="llama_generation", 
    model=model,
    labels={"framework": "transformers", "type": "causal-lm"},  
    metadata={
        "framework_version": "transformers-4.x",
        "tokenizer_name": MODEL_DIR,
    },
)
print(f"Model successfully saved in BentoML model store: {model_ref}")

tokenizer_path = model_ref.path_of("tokenizer")

# Save the tokenizer to the designated tokenizer path. This ensures that when the model is loaded 
# from BentoML, the tokenizer is also readily available, making it trivial to prepare inputs for the model.
tokenizer.save_pretrained(tokenizer_path)
print(f"Tokenizer saved successfully at: {tokenizer_path}")