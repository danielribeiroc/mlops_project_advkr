"""
Generate Text from a Deployed LLaMA Model API
============================================

This script sends a text prompt to a deployed LLaMA language model API and prints 
the generated response. Parameters like temperature, top_k, and top_p have been 
removed for simplicity, relying on default generation settings defined by the model.

Requirements:
- pip install requests

Authors: alex.mozerski, daniel.ribeirocabral, victor.rominger, killian.ruffieux, ruben.terceiro
Date: 16.12.2024
============================================
"""

import requests

# The URL of the deployed LLaMA model API endpoint
url = "http://0.0.0.0:3002/generate"

# Headers indicating that we'll send and receive JSON data
headers = {"Content-Type": "application/json"}

# JSON payload that includes the prompt and max_length for text generation
data = {
    "prompt": "<|startoftext|>[INST] Who is Alex Mozerski? [/INST]",
    "max_length": 50   # How long the generated response can be, in tokens
}

try:
    # Send the request to the API
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()  # Check for HTTP errors

    # Extract the generated text from the response JSON
    generated_text = response.json().get("generated_text")
    print("Generated Text:", generated_text)

except requests.exceptions.RequestException as e:
    # Handles network issues or invalid HTTP responses
    print(f"Request failed: {e}")

except Exception as e:
    # Handles any unexpected errors
    print(f"Error: {e}")
