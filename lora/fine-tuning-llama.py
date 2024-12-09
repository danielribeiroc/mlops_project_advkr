"""
Fine-tuning a LLaMA Language Model with LoRA (Low-Rank Adaptation)
==================================================================

This script fine-tunes a pre-trained LLaMA language model using LoRA (Low-Rank Adaptation)
on a custom dataset of local text files. The fine-tuned model and tokenizer are then saved for later use.

Requirements:
- Install the necessary packages: transformers, peft, datasets, torch, huggingface_hub, yaml, bentoml, tqdm

Module : MLOps
Authors: alex.mozerski, daniel.ribeirocabral, victor.rominger, killian.ruffieux, ruben.terceiro
Date: 05.12.2024
===================================================================
"""

# Import necessary libraries
import os
import yaml
import torch
import json
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
from huggingface_hub import login
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

def configure_lora(model):
    """
    Configure Low-Rank Adaptation (LoRA) for the given model.

    This function sets up the LoRA configuration with specified hyperparameters,
    applies it to the model, freezes the base model parameters, and sets
    the LoRA parameters to require gradients.

    Parameters:
        model (transformers.PreTrainedModel): The pre-trained model to configure with LoRA.

    Returns:
        transformers.PreTrainedModel: The model configured with LoRA.
    """
    # User input for LoRA hyperparameters
    # Uncomment the lines below to enable dynamic configuration
    # r = int(input("Enter rank (r): "))
    # lora_alpha = int(input("Enter LoRA alpha: "))
    # lora_dropout = float(input("Enter LoRA dropout (0-1): "))
    # target_modules = input("Enter target modules (comma-separated, e.g., 'q_proj,v_proj'): ").split(",")

    # LoRA hyperparameters
    r = 64  # Rank of the decomposition
    lora_alpha = 64  # Scaling factor
    lora_dropout = 0.1  # Dropout rate
    target_modules = ["q_proj", "v_proj"]  # Modules to apply LoRA to

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"Applying LoRA configuration: {lora_config}")

    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)

    # Freeze base model parameters
    for param in lora_model.base_model.parameters():
        param.requires_grad = False

    # Enable gradients for LoRA parameters
    for name, param in lora_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            print(f"LoRA Parameter: {name} will be fine-tuned.")

    return lora_model

def load_and_tokenize_local_files(tokenizer, folder_path):
    """
    Load and tokenize text data from local files.

    This function reads all text files in the specified folder, tokenizes the text,
    and prepares it for training.

    Parameters:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        folder_path (str): Path to the folder containing text files.

    Returns:
        datasets.Dataset: The tokenized dataset.
    """
    print("Loading and tokenizing local text files...")

    # Read all text files from the folder
    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())

    # Combine and split text into lines
    combined_text = "\n".join(texts)
    lines = [line.strip() for line in combined_text.split("\n") if line.strip()]

    # Create dataset from text lines
    dataset = Dataset.from_dict({"text": lines})

    # Tokenization function
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        tokens["labels"] = tokens["input_ids"].copy()
        
        # Replace padding token IDs in labels with -100 to ignore in loss computation
        tokens["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in tokens["labels"]
        ]
        return tokens

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print("Local dataset tokenized successfully.")
    return tokenized_dataset

def create_dataloader(tokenized_dataset, batch_size=4):
    """
    Create a DataLoader from the tokenized dataset.

    Parameters:
        tokenized_dataset (datasets.Dataset): The tokenized dataset.
        batch_size (int): The batch size for training.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for training.
    """
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

def setup_training(model):
    """
    Set up the optimizer and learning rate scheduler.

    Parameters:
        model (transformers.PreTrainedModel): The model to train.

    Returns:
        tuple: The optimizer and scheduler.
    """
    # Filter parameters that require gradients
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10000
    )
    return optimizer, scheduler

def train_model(model, data_loader, optimizer, scheduler, epochs=3):
    """
    Train the model.

    Parameters:
        model (transformers.PreTrainedModel): The model to train.
        data_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (transformers.Scheduler): The learning rate scheduler.
        epochs (int): Number of training epochs.
    """
    print(f"Starting training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = []
        for batch in tqdm(data_loader):
            # Move batch to device
            inputs = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Record loss
            epoch_losses.append(loss.item())

        # Log epoch loss
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

def save_model(model, tokenizer, model_dir="fine_tuned_lora_llama"):
    """
    Save the fine-tuned model and tokenizer.

    Parameters:
        model (transformers.PreTrainedModel): The trained model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        model_dir (str): Directory to save the model and tokenizer.
    """
    print(f"Saving model to {model_dir}...")
    os.makedirs(model_dir, exist_ok=True)

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    # Resize the model's embeddings to match the tokenizer
    #model.resize_token_embeddings(len(tokenizer)) 

    # Remove PEFT configuration from the model's config
    #if hasattr(model.config, "peft_config"):
    #    del model.config.peft_config

    # Ensure the vocab_size is correct
    model.config.vocab_size = len(tokenizer)

    # Optional: Verify the vocab_size
    print(f"Model config vocab_size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model embeddings size: {model.get_input_embeddings().weight.shape}")
    
    # Save the model
    model.save_pretrained(model_dir)

    # Handle potential torch.dtype in tokenizer init kwargs
    if hasattr(tokenizer, "init_kwargs"):
        for key, value in tokenizer.init_kwargs.items():
            if isinstance(value, torch.dtype):
                tokenizer.init_kwargs[key] = str(value)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save the tokenizer
    tokenizer.save_pretrained(model_dir)
    print("Model and tokenizer saved successfully.")

# Main execution starts here
if __name__ == "__main__":
    # Change to the working directory @TODO Delete
    # os.chdir('/home/mozerski/TM-2024/')  # Update with your working directory

    # Load Hugging Face token from YAML file and log in
    with open("hf_token.yaml", "r") as file:
        token_config = yaml.safe_load(file)
    token = token_config.get("hf_token", None)

    if token:
        login(token=token)
        print("Hugging Face token loaded and logged in successfully.")
    else:
        print("No Hugging Face token found in hf_token.yaml.")
        exit(1)

    # Specify the model name
    model_name = "meta-llama/Llama-3.2-1B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=token,
        torch_dtype=torch.float16
    )

    # Assign padding token if not set
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token. Assigning pad_token as eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens and resize tokenizer
    special_tokens_dict = {
        'additional_special_tokens': ['<|startoftext|>', '[INST]', '[/INST]', '<|endoftext|>']
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")

    # Resize tokenizer embeddings
    tokenizer.model_max_length = 512  # Ensure tokenizer max length is set
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=token,
        torch_dtype=torch.float16
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # Optionally prepare model for k-bit training
    # Uncomment the line below if using quantization-aware training
    # base_model = prepare_model_for_kbit_training(base_model)

    # Configure LoRA on the base model
    model = configure_lora(base_model)

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model and tokenizer loaded on {device}.")

    # Load and tokenize the local dataset
    folder_path = "./DATASET"
    tokenized_dataset = load_and_tokenize_local_files(tokenizer, folder_path)

    # Create DataLoader
    data_loader = create_dataloader(tokenized_dataset, batch_size=8)

    # Set up optimizer and scheduler
    optimizer, lr_scheduler = setup_training(model)

    # Train the model
    train_model(model, data_loader, optimizer, lr_scheduler, epochs=10)

    # Save the trained model and tokenizer
    save_model(model, tokenizer)
