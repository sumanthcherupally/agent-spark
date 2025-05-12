from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

class SWEBenchModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
    ):
        """
        Initialize the SWE-bench model.
        
        Args:
            model_name: Name or path of the pre-trained model
            device: Device to run the model on
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Add special tokens if needed
        special_tokens = {
            "additional_special_tokens": [
                "<issue>",
                "<code>",
                "<solution>",
                "<test>",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_input(
        self,
        issue_description: str,
        code_context: str,
        test_cases: Optional[List[str]] = None,
    ) -> str:
        """
        Prepare the input prompt for the model.
        
        Args:
            issue_description: Description of the GitHub issue
            code_context: Relevant code context
            test_cases: Optional test cases for verification
            
        Returns:
            Formatted input prompt
        """
        prompt = f"<issue>{issue_description}</issue>\n"
        prompt += f"<code>{code_context}</code>\n"
        
        if test_cases:
            prompt += f"<test>{' '.join(test_cases)}</test>\n"
            
        prompt += "<solution>"
        return prompt

    def generate_solution(
        self,
        issue_description: str,
        code_context: str,
        test_cases: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate a solution for the given issue.
        
        Args:
            issue_description: Description of the GitHub issue
            code_context: Relevant code context
            test_cases: Optional test cases for verification
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated solution
        """
        prompt = self.prepare_input(issue_description, code_context, test_cases)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )
        
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the solution part
        solution = solution.split("<solution>")[-1].strip()
        
        return solution

    def save_model(self, path: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None):
        """Load a saved model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        model = cls.__new__(cls)
        model.device = device
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.model = AutoModelForCausalLM.from_pretrained(path).to(device)
        return model 