import os
import boto3
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import pipeline
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

DATA_PATH = ("../data")

MODEL_ID = "meta-llama/Llama-3.2-1B"

def read_file_content(source, path):
    """
    Reads the content of a file based on its source (local or S3).

    Args:
        source (str): "local" or "s3".
        path (str): File path (for local) or bucket/key (for S3).

    Returns:
        str: The content of the file.
    """
    if source == "local":
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    elif source == "s3":
        bucket, key = path.split("/", 1)
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")
    else:
        raise ValueError("Invalid source. Use 'local' or 's3'.")

def load_files_to_dict(base_path, source="local", prefix=""):
    """
    Loads text files from a source into a dictionary.

    Args:
        source (str): "local" or "s3".
        base_path (str): Local folder path or S3 bucket name.
        prefix (str): Prefix for filtering files in S3 (ignored for local).

    Returns:
        dict: A dictionary with filenames as keys and file contents as values.
    """
    files_dict = {}

    if source == "local":
        for filename in os.listdir(base_path):
            file_path = os.path.join(base_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                files_dict[filename] = read_file_content(source, file_path)

    elif source == "s3":
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=base_path, Prefix=prefix)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".txt"):
                files_dict[key] = read_file_content(source, f"{base_path}/{key}")
    else:
        raise ValueError("Invalid source. Use 'local' or 's3'.")

    return files_dict

def init_pipe(model_id, model_cache_dir="./models"):
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_cache_dir)
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype="float16",
    )
    print("Model loaded")

    return pipe

def init_retriever(documents):

    texts = [content for content in documents.values()]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_texts(texts, embeddings)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "lambda_mult": 0.5}
    )

    return retriever

def rag_request(query, retriever, pipe):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    ##pipe.model.to(device)
    ##pipe.tokenizer.to(device)

    docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = pipe(prompt, max_new_tokens=80, do_sample=True, temperature=0.7)
    generated_text = response[0]["generated_text"]
    answer = generated_text.split("Answer:")[-1].strip()

    return answer

