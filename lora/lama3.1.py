# %%
import yaml
from huggingface_hub import login
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import matplotlib.pyplot as plt
import PyPDF2
from tqdm import tqdm

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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    torch_dtype=torch.float16,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model and tokenizer loaded.")

def configure_lora():
    
    #r = int(input("Enter rank (r): "))
    #lora_alpha = int(input("Enter LoRA alpha: "))
    #lora_dropout = float(input("Enter LoRA dropout (0-1): "))
    #target_modules = input("Enter target modules (comma-separated, e.g., 'q_proj,v_proj'): ").split(",")

    r = 4
    lora_alpha = 8
    lora_dropout = 0.1
    target_modules = ['q_proj', 'v_proj']

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    print(f"Applying LoRA configuration: {lora_config}")
    return get_peft_model(model, lora_config)

def load_and_tokenize_wikitext(tokenizer):
    print("Loading and tokenizing Wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Tokenize dataset and return input IDs and attention mask
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Ensure the dataset returns a dictionary
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    print("Dataset tokenized successfully.")
    return DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

# Training setup
def setup_training(model):
    """
    Set up optimizer and learning rate scheduler.
    """
    optimizer = AdamW(model.parameters(), lr=5e-4)
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=10000
    )
    return optimizer, scheduler

# Training loop
def train_model(model, data_loader, optimizer, scheduler, epochs=1):
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
            if outputs.loss is None:
                print("Skipping batch due to None loss.")
                continue
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
    
    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid()
    plt.show()

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

model = configure_lora()
print("LoRA configuration applied successfully!")
data_loader = load_and_tokenize_wikitext(tokenizer)
optimizer, lr_scheduler = setup_training(model)
train_model(model, data_loader, optimizer, lr_scheduler)
save_model(model, tokenizer)

