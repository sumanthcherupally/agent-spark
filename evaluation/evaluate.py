import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from models.model import SWEBenchModel
from utils.data_utils import load_swe_bench_dataset
from utils.metrics import compute_metrics, compute_test_pass_rate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(
    model: SWEBenchModel,
    test_data: List[Dict],
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: The trained model
        test_data: List of test samples
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.model.eval()
    all_predictions = []
    all_references = []
    all_test_cases = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i + batch_size]
        
        # Generate predictions
        predictions = []
        for sample in batch:
            prediction = model.generate_solution(
                sample["issue_description"],
                sample["code_context"],
                sample.get("test_cases", []),
            )
            predictions.append(prediction)
        
        # Collect results
        all_predictions.extend(predictions)
        all_references.extend([sample["solution"] for sample in batch])
        all_test_cases.extend([sample.get("test_cases", []) for sample in batch])
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    
    # Compute test pass rate if test cases are available
    if any(all_test_cases):
        test_pass_rate = compute_test_pass_rate(
            all_predictions,
            all_references,
            all_test_cases,
        )
        metrics["test_pass_rate"] = test_pass_rate
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_file", type=str, help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Load model
    model = SWEBenchModel.load_model(args.model_path)
    
    # Load test data
    test_data = load_swe_bench_dataset(args.test_data)
    
    # Evaluate model
    metrics = evaluate_model(model, test_data, args.batch_size)
    
    # Log metrics
    logger.info("Evaluation metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")

if __name__ == "__main__":
    main() 