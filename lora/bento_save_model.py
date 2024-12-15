"""
Deploying and Saving a Fine-tuned LLaMA Language Model with BentoML
===================================================================

This script loads a fine-tuned LLaMA language model and its tokenizer, then saves
them to the BentoML model store for later deployment and usage. The model is
configured to run on GPU if available, otherwise it defaults to CPU.

Requirements:
- Install the necessary packages: transformers, bentoml, torch

Module : MLOps
Authors: alex.mozerski, daniel.ribeirocabral, victor.rominger, killian.ruffieux, ruben.terceiro
Date: 05.12.2024
====================================================================
"""
import bentoml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

BENTOML_STORAGE_PATH = "/actions-runner/bentoml"
os.environ["BENTOML_HOME"] = BENTOML_STORAGE_PATH

# Define the directory where the fine-tuned model is saved and determine the device
MODEL_DIR = "fine_tuned_lora_llama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and model...")
# Load the tokenizer from the specified model directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print("Tokenizer and model loaded successfully.")

print("Saving model to BentoML model store...")
# Save the PyTorch model to BentoML's model store
model_ref = bentoml.pytorch.save_model(
    name="llama_generation",
    model=model,
    labels={"framework": "transformers", "type": "causal-lm"},
    metadata={
        "framework_version": "transformers-4.x",
        "tokenizer_name": MODEL_DIR,
    },
)
print(f"Model saved successfully in BentoML model store: {model_ref}")
# Retrieve the path where the tokenizer should be saved within the BentoML model store

tokenizer_path = model_ref.path_of("tokenizer")
tokenizer.save_pretrained(tokenizer_path)
print(f"Tokenizer saved successfully at: {tokenizer_path}")
