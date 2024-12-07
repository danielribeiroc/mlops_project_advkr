"""
Generating Text Using the Deployed LLaMA Model API
==================================================

This script sends a request to the deployed LLaMA language model API to generate text based
on a provided prompt. It constructs the necessary JSON payload with parameters for text
generation, handles the API response, and manages potential request errors.

Requirements:
- Install the necessary package: requests

Module : MLOps
Authors: alex.mozerski, daniel.ribeirocabral, victor.rominger, killian.ruffieux, ruben.terceiro
Date: 05.12.2024
===================================================
"""

import requests

# Define the API endpoint URL and headers
url = "http://0.0.0.0:3002/generate"
headers = {"Content-Type": "application/json"}

# Prepare the data payload with prompt and generation parameters
data = {
    "prompt": "<|startoftext|>[INST] Who is Alex Mozerski? [/INST]",
    "max_length": 100,     # Maximum length of the generated text
    "temperature": 0.9,    # Sampling temperature for controlling randomness
    "top_k": 50,           # Top-k sampling parameter for selecting likely tokens
    "top_p": 0.9,          # Nucleus sampling parameter for cumulative probability
}

try:
    # Send a POST request to the API with the JSON payload
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP error responses
    
    # Extract and print the generated text from the API response
    generated_text = response.json().get("generated_text")
    print("Generated Text:", generated_text)
    
except requests.exceptions.RequestException as e:
    # Handle any request-related errors (e.g., network issues, invalid responses)
    print(f"Request failed: {e}")
    
except Exception as e:
    # Handle any other unforeseen errors
    print(f"Error: {e}")