import torch
from datasets import load_dataset
import wandb
from tqdm import trange
from typing import List
import random
from .ppo_lm import PPOLM

def train_lm(
    dataset_name: str = "imdb",          
    split: str = "train",
    text_column: str = "text",           
    num_samples: int = 5000,            
    epochs: int = 5,                    
    batch_size: int = 2,                
    max_new_tokens: int = 48,
    gradient_accumulation_steps: int = 8
):
    wandb.init(project="rl-study-ppo-lm")

    print(f"Loading dataset: {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, split=split)

    # For IMDb: Use only positive reviews (label == 1) as prompts for positive generation
    if dataset_name == "imdb":
        dataset = dataset.filter(lambda x: x["label"] == 1)  # Positive reviews only
        data_slice = dataset[:num_samples]
        prompts: List[str] = [text[:200] for text in data_slice["text"]]  # Truncate to save memory
    else:
        # Generic: Take the specified text column
        data_slice = dataset[:num_samples]
        prompts: List[str] = [text[:200] for text in data_slice[text_column]]

    # Shuffle for better training
    random.shuffle(prompts)

    print(f"Loaded {len(prompts)} prompts. Example: {prompts[0][:200]}...")

    model = PPOLM(model_name="distilgpt2", use_amp=True)  # Uses distilgpt2 by default

    global_step = 0
    for ep in trange(epochs, desc="PPO-LM Epochs"):
        random.shuffle(prompts)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            loss = model.update(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            epoch_loss += loss
            num_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                wandb.log({"global_step": global_step, "loss": loss})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {ep + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # Save the full model (including value head) and keep a model-only copy for compatibility
    save_path_full = "models/ppo_lm_distilgpt2_full.pt"
    save_path_model = "models/ppo_lm_distilgpt2.pt"
    torch.save({'model': model.model.state_dict(), 'value_head': model.value_head.state_dict()}, save_path_full)
    torch.save(model.model.state_dict(), save_path_model)
    print(f"Full model saved to {save_path_full} (includes value head)")
    print(f"Model-only weights saved to {save_path_model}")

if __name__ == "__main__":
    # Train on positive IMDb reviews → encourages positive language generation
    train_lm(dataset_name="imdb", num_samples=5000, epochs=5, batch_size=2, gradient_accumulation_steps=8)

    # Alternatives:
    # train_lm(dataset_name="allenai/real-toxicity-prompts", text_column="prompt", num_samples=5000)  # For detox
    # train_lm(dataset_name="HuggingFaceH4/cherry_picked_prompts", text_column="prompt", num_samples=2000)