"""
Model wrappers for TinyZero
Handles policy and reference models
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import os

# Disable MPS (Apple Silicon GPU) - force CPU only
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class PolicyModel:
    """Wrapper for policy model (the one we train)"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
        """
        # Force CPU if not CUDA
        if device != "cuda" or not torch.cuda.is_available():
            device = "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model WITHOUT device_map to avoid MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Always float32 for CPU
            low_cpu_mem_usage=True
        )
        
        # Manually move to device
        self.model = self.model.to(device)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
    
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 1.0,        
        top_k: int = 0             
    ) -> List[str]:
        """
        Generate responses for prompts
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            top_p: Nucleus sampling parameter (1.0 = disabled)
            top_k: Top-k sampling parameter (0 = disabled)
        
        Returns:
            List of generated texts
        """
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p if top_p < 1.0 else None,      
                top_k=top_k if top_k > 0 else None,        
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Remove prompt from generated text
        responses = []
        for prompt, generated in zip(prompts, generated_texts):
            # Remove the prompt part
            if generated.startswith(prompt):
                response = generated[len(prompt):].strip()
            else:
                response = generated.strip()
            responses.append(response)
        
        return responses
    
    def get_log_probs(
        self,
        prompts: List[str],
        completions: List[str]
    ) -> torch.Tensor:
        """
        Get log probabilities for completions given prompts
        
        Args:
            prompts: List of prompts
            completions: List of completions
        
        Returns:
            Log probabilities tensor
        """
        # Combine prompts and completions
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        # Tokenize
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get log probs for actual tokens
        # Simplified - return mean log prob
        return log_probs.mean(dim=-1).mean(dim=-1)
    
    def parameters(self):
        """Return model parameters for optimization"""
        return self.model.parameters()
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()


class ReferenceModel:
    """Frozen reference model for computing V*"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
        """
        # Force CPU if not CUDA
        if device != "cuda" or not torch.cuda.is_available():
            device = "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model WITHOUT device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Always float32 for CPU
            low_cpu_mem_usage=True
        )
        
        # Manually move to device
        self.model = self.model.to(device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 1.0,
        num_samples: int = 1,
        top_p: float = 1.0,
        top_k: int = 0 
    ) -> List[List[str]]:
        """
        Generate multiple samples per prompt (for V* computation)
        
        Args:
            prompts: List of prompts
            max_length: Max generation length
            temperature: Sampling temperature
            num_samples: Number of samples per prompt
        
        Returns:
            List of lists of generated texts
        """
        all_samples = []
        
        for prompt in prompts:
            samples = []
            
            # Generate num_samples completions
            for _ in range(num_samples):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_p=top_p if top_p < 1.0 else None,
                        top_k=top_k if top_k > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # Remove prompt
                if generated.startswith(prompt):
                    response = generated[len(prompt):].strip()
                else:
                    response = generated.strip()
                samples.append(response)
            
            all_samples.append(samples)
        
        return all_samples


# Test models
if __name__ == "__main__":
    print("Testing model loading...")
    
    # Test policy model
    policy = PolicyModel("gpt2", device="cpu")
    
    test_prompt = ["What is 5 + 3?"]
    response = policy.generate(test_prompt, max_length=50)
    
    print(f"Prompt: {test_prompt[0]}")
    print(f"Response: {response[0]}")
    
    print("\nModel loading successful!")