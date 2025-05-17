import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm

from models.model import SWEBenchModel
from utils.data_utils import load_swe_bench_dataset, prepare_batch
from utils.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(
    model: SWEBenchModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    warmup_steps: int,
    max_grad_norm: float,
    output_dir: str,
    use_wandb: bool = True,
):
    """
    Train the SWE-bench model.
    
    Args:
        model: The model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        output_dir: Directory to save the model
        use_wandb: Whether to use Weights & Biases for logging
    """
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    if use_wandb:
        wandb.init(project="swe-bench-model")
    
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            inputs = prepare_batch(batch, model.tokenizer, model.device)
            
            outputs = model.model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            if use_wandb:
                wandb.log({"train_loss": loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        val_metrics = evaluate(model, val_dataloader)
        logger.info(f"Validation metrics: {val_metrics}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                **val_metrics
            })
        
        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            model.save_model(os.path.join(output_dir, "best_model"))
            logger.info("Saved new best model!")

def evaluate(model: SWEBenchModel, dataloader: DataLoader) -> Dict[str, float]:
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = prepare_batch(batch, model.tokenizer, model.device)
            
            outputs = model.model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Generate predictions
            predictions = model.generate_solution(
                batch["issue_description"],
                batch["code_context"],
                batch.get("test_cases"),
            )
            
            all_predictions.extend(predictions)
            all_references.extend(batch["solution"])
    
    metrics = compute_metrics(all_predictions, all_references)
    metrics["val_loss"] = total_loss / len(dataloader)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train SWE-bench model")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the pre-trained model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = SWEBenchModel(args.model_name)
    
    # Load datasets
    train_dataset = load_swe_bench_dataset(args.train_data)
    val_dataset = load_swe_bench_dataset(args.val_data)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,  # Custom collate function will be applied in prepare_batch
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    
    # Train model
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )

if __name__ == "__main__":
    main() 