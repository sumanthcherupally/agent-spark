import json
from typing import Dict, List, Optional, Union
import torch
from transformers import PreTrainedTokenizer

def load_swe_bench_dataset(file_path: str) -> List[Dict]:
    """
    Load SWE-bench dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        List of dictionaries containing the dataset samples
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def prepare_batch(
    batch: List[Dict],
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of data for model input.
    
    Args:
        batch: List of dictionaries containing the batch samples
        tokenizer: Tokenizer to use for encoding
        device: Device to move tensors to
        
    Returns:
        Dictionary of tensors ready for model input
    """
    # Extract fields from batch
    issue_descriptions = [sample["issue_description"] for sample in batch]
    code_contexts = [sample["code_context"] for sample in batch]
    solutions = [sample["solution"] for sample in batch]
    test_cases = [sample.get("test_cases", []) for sample in batch]
    
    # Prepare input prompts
    prompts = []
    for issue, code, tests in zip(issue_descriptions, code_contexts, test_cases):
        prompt = f"<issue>{issue}</issue>\n"
        prompt += f"<code>{code}</code>\n"
        if tests:
            prompt += f"<test>{' '.join(tests)}</test>\n"
        prompt += "<solution>"
        prompts.append(prompt)
    
    # Tokenize inputs
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )
    
    # Tokenize solutions
    solution_inputs = tokenizer(
        solutions,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    
    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    solution_inputs = {k: v.to(device) for k, v in solution_inputs.items()}
    
    # Prepare labels
    labels = solution_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
    
    return {
        **inputs,
        "labels": labels,
    }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of dictionaries containing the batch samples
        
    Returns:
        Dictionary of tensors ready for model input
    """
    # This is a placeholder - the actual collation is done in prepare_batch
    return batch 