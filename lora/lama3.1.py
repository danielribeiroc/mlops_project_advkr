# %%
import yaml
from huggingface_hub import login
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from datasets import Dataset
from peft import PeftModel

# Configure LoRA
def configure_lora(model):

    #r = int(input("Enter rank (r): "))
    #lora_alpha = int(input("Enter LoRA alpha: "))
    #lora_dropout = float(input("Enter LoRA dropout (0-1): "))
    #target_modules = input("Enter target modules (comma-separated, e.g., 'q_proj,v_proj'): ").split(",")

    r = 64
    lora_alpha = 64
    lora_dropout = 0.1
    target_modules = ["q_proj","v_proj"]#,"o_proj"] #, 'k_proj', 'output_proj']

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"Applying LoRA configuration: {lora_config}")
    lora_model = get_peft_model(model, lora_config)

    # Freeze Base Model Parameters
    for param in lora_model.base_model.parameters():
        param.requires_grad = False
        
    # Set Requires Grad for LoRA Parameters
    for name, param in lora_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            print(f"LoRA Parameter: {name} will be fine-tuned.")

    return lora_model
    

# Load and tokenize local files
def load_and_tokenize_local_files(tokenizer, folder_path):
    print("Loading and tokenizing local text files...")

    # Read all text files from the folder
    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())

    # Combine all text data into a single string
    combined_text = "\n".join(texts)

    # Split combined text into lines
    lines = [line.strip() for line in combined_text.split("\n") if line.strip()]

    # Create dataset with single sequences (prompt + response)
    dataset = Dataset.from_dict({"text": lines})

    # Tokenize the dataset
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        tokens["labels"] = tokens["input_ids"].copy()
        # Replace padding token id's in labels by -100 so they are ignored in loss computation
        tokens["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in tokens["labels"]
        ]
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print("Local dataset tokenized successfully.")
    return tokenized_dataset


# DataLoader Creation
def create_dataloader(tokenized_dataset, batch_size=4):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)


# Training setup
def setup_training(model):
    """
    Set up optimizer and learning rate scheduler.
    """
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=10000
    )
    return optimizer, scheduler
    
# Training loop
def train_model(model, data_loader, optimizer, scheduler, epochs=3):
    print(f"Starting training for {epochs} epochs...")
    model.train()
    losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = []
        for batch in tqdm(data_loader):
            inputs = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

def save_model(model, tokenizer, model_dir="fine_tuned_lora_llama"):
    """
    Save the model and tokenizer.
    """
    print(f"Saving model to {model_dir}...")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model.save_pretrained(model_dir)
    
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

os.chdir('/home/mozerski/TM-2024/')

with open("hf_token.yaml", "r") as file:
    token_config = yaml.safe_load(file)
token = token_config.get("hf_token", None)
login(token=token)

if token:
    print("Hugging Face token loaded and login successfully.")
else:
    print("No Hugging Face token found in hf_token.yaml.")

model_name = "meta-llama/Llama-3.2-1B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                        token=token,
                                        torch_dtype=torch.float16)
# define the pad token
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad_token. Assigning pad_token as eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

# Add Special Tokens and Resize Tokenizer
special_tokens_dict = {'additional_special_tokens': ['<|startoftext|>', '[INST]', '[/INST]', '<|endoftext|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens.")

# Load model and prepare for LoRA
base_model = AutoModelForCausalLM.from_pretrained(model_name, token=token, torch_dtype=torch.float16)
base_model.resize_token_embeddings(len(tokenizer))

#base_model = prepare_model_for_kbit_training(base_model)

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)

print("Model and tokenizer loaded.")

folder_path = "./DATASET"
tokenized_dataset = load_and_tokenize_local_files(tokenizer, folder_path)
data_loader = create_dataloader(tokenized_dataset, batch_size=8)

model = configure_lora(base_model)
optimizer, lr_scheduler = setup_training(model)
train_model(model, data_loader, optimizer, lr_scheduler, epochs=10)

# Save the trained and merged model
save_model(model, tokenizer)



