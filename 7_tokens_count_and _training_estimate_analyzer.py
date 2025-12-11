#!/usr/bin/env python3
"""
Script to calculate the number of tokens in a text file.
For use with transformer models like DistilGPT-2.
"""

from transformers import AutoTokenizer
import argparse
import os

def count_tokens(file_path, model_name="distilgpt2"):
    """
    Count tokens in a text file using a specific tokenizer.
    
    Args:
        file_path: Path to the text file
        model_name: Hugging Face model name for the tokenizer
        
    Returns:
        Tuple of (total_tokens, num_lines, avg_tokens_per_line)
    """
    
    # Load the tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present (for GPT-2 models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    total_tokens = 0
    num_lines = 0
    
    # Read and process the file
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                tokens = tokenizer.encode(line, add_special_tokens=False)
                total_tokens += len(tokens)
                num_lines += 1
    
    # Calculate average tokens per line
    avg_tokens_per_line = total_tokens / num_lines if num_lines > 0 else 0
    
    return total_tokens, num_lines, avg_tokens_per_line

def estimate_training_steps(total_tokens, batch_size=4, seq_length=512, epochs=3):
    """
    Estimate training steps based on token count.
    
    Args:
        total_tokens: Total tokens in training data
        batch_size: Training batch size
        seq_length: Sequence length for training
        epochs: Number of training epochs
        
    Returns:
        Estimated number of training steps
    """
    # Calculate tokens per batch
    tokens_per_batch = batch_size * seq_length
    
    # Calculate steps per epoch
    steps_per_epoch = total_tokens // tokens_per_batch
    if total_tokens % tokens_per_batch > 0:
        steps_per_epoch += 1
    
    # Total steps for all epochs
    total_steps = steps_per_epoch * epochs
    
    return total_steps

def debug_long_lines(file_path, max_samples=3):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        tokens = tokenizer.encode(line.strip(), add_special_tokens=False)
        if len(tokens) > 1024:  # Show really long lines
            print(f"Line {i}: {len(tokens)} tokens")
            print(f"First 100 chars: {line[:100]}...")
            print(f"Last 100 chars: {line[-100:]}...")
            print("-" * 50)
            
            if i >= max_samples:
                break

# Run it on your file


def main():
    parser = argparse.ArgumentParser(description="Count tokens in a text file")
    parser.add_argument("file_path", help="Path to the text file")
    parser.add_argument("--model", default="distilgpt2", 
                       help="Hugging Face model name (default: distilgpt2)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training estimation (default: 4)")
    parser.add_argument("--seq_length", type=int, default=512,
                       help="Sequence length for training (default: 512)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found!")
        return
    
    # Count tokens
    total_tokens, num_lines, avg_tokens = count_tokens(args.file_path, args.model)
    
    # Print results
    print("\n" + "="*50)
    print("TOKEN COUNT RESULTS")
    print("="*50)
    print(f"Model/tokenizer: {args.model}")
    print(f"File: {args.file_path}")
    print(f"Lines processed: {num_lines:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per line: {avg_tokens:.1f}")
    
    # Estimate training steps
    if total_tokens > 0:
        total_steps = estimate_training_steps(
            total_tokens, args.batch_size, args.seq_length, args.epochs
        )
        print(f"\nTRAINING ESTIMATION (batch_size={args.batch_size}, seq_length={args.seq_length})")
        print(f"Steps per epoch: ~{estimate_training_steps(total_tokens, args.batch_size, args.seq_length, 1):,}")
        print(f"Total steps for {args.epochs} epochs: ~{total_steps:,}")
    
    print("="*50)
if __name__ == "__main__":
    main()