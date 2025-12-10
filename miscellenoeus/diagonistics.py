# diagnostic.py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

with open("bmw_press_datasets/chunked/train_chunked.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total lines in file: {len(lines)}")

# Check first 5 lines' token lengths
for i, line in enumerate(lines[:30]):
    tokens = tokenizer.encode(line, add_special_tokens=False)
    print(f"Line {i}: {len(tokens)} tokens")
    
    if len(tokens) > 900:
        print(f"  ⚠️ WARNING: Line is {len(tokens)} tokens!")
        print(f"  First 50 chars: '{line[:50]}...'")