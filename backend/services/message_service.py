from models.message import Message
from datetime import datetime
from services.rag_functions import init_retriever, load_files_to_dict, rag_request
import requests

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

    url = "http://0.0.0.0:3002/generate" # TODO
    headers = {"Content-Type": "application/json"}

    data = {
        "prompt": f"<|startoftext|>[INST] {message.text} [/INST]",
        "max_length": 100, 
        "temperature": 0.9, 
        "top_k": 50,     
        "top_p": 0.9,        
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        generated_text = response.json().get("generated_text")
        return {"sender": "lora", "text": generated_text, "timestamp": datetime.utcnow()}
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    except Exception as e:
        print(f"Error: {e}")

