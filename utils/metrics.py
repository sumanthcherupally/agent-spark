from typing import Dict, List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import Levenshtein

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model predictions.
    
    Args:
        predictions: List of predicted solutions
        references: List of reference solutions
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Compute exact match accuracy
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    exact_match_accuracy = exact_matches / len(predictions)
    
    # Compute Levenshtein distance-based metrics
    distances = [Levenshtein.distance(p.strip(), r.strip()) for p, r in zip(predictions, references)]
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    
    # Compute token-level precision, recall, and F1
    token_predictions = [p.split() for p in predictions]
    token_references = [r.split() for r in references]
    
    # Flatten token lists for sklearn metrics
    flat_preds = [token for tokens in token_predictions for token in tokens]
    flat_refs = [token for tokens in token_references for token in tokens]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_refs,
        flat_preds,
        average="weighted",
        zero_division=0,
    )
    
    return {
        "exact_match_accuracy": exact_match_accuracy,
        "avg_levenshtein_distance": avg_distance,
        "max_levenshtein_distance": max_distance,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }

def compute_test_pass_rate(
    predictions: List[str],
    references: List[str],
    test_cases: List[List[str]],
) -> float:
    """
    Compute the rate at which the model's solutions pass the test cases.
    
    Args:
        predictions: List of predicted solutions
        references: List of reference solutions
        test_cases: List of test cases for each solution
        
    Returns:
        Test pass rate (percentage of solutions that pass all test cases)
    """
    # This is a placeholder - in practice, you would need to:
    # 1. Execute the predicted solutions in a safe environment
    # 2. Run the test cases against the executed solutions
    # 3. Count how many solutions pass all their test cases
    
    # For now, we'll return a dummy value
    return 0.0 