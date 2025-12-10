#!/usr/bin/env python3
"""
BMW Press Release Chunker - Clean Version
Processes augmented files and creates clean chunks for training
"""

import os
import json
import math
from pathlib import Path
from transformers import AutoTokenizer

# Configuration
INPUT_DIR = "bmw_press_datasets/augmented"
OUTPUT_DIR = "bmw_press_datasets/chunked"
MODEL_NAME = "distilgpt2"  # or "gpt2", "distilgpt2", "facebook/opt-125m"
MAX_TOKENS = 512  # Conservative for small models
OVERLAP_TOKENS = 128  # Overlap between chunks

def setup_directories():
    """Create output directory"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output will be saved to: {OUTPUT_DIR}")

def load_tokenizer():
    """Load tokenizer for the target model"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer: {MODEL_NAME}")
    print(f"   Context: {MAX_TOKENS} tokens, Overlap: {OVERLAP_TOKENS} tokens")
    return tokenizer

def split_into_press_releases(content):
    """Split by 80 equals signs to get individual press releases"""
    releases = []
    parts = content.split("=" * 80)
    
    for part in parts:
        part = part.strip()
        if part and len(part) > 100:  # Skip empty or very short
            releases.append(part)
    
    return releases

def create_clean_chunks(text, tokenizer, chunk_id=None):
    """
    Create clean chunks from text without any metadata
    Returns list of chunk texts
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # If it fits in one chunk, return as is
    if len(tokens) <= MAX_TOKENS:
        return [text]
    
    # Calculate chunking parameters
    stride = MAX_TOKENS - OVERLAP_TOKENS  # How far to move each window
    chunks = []
    
    # Create overlapping chunks
    for i in range(0, len(tokens), stride):
        # Get chunk tokens
        chunk_start = i
        chunk_end = min(i + MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[chunk_start:chunk_end]
        
        # Skip very short final chunks
        if len(chunk_tokens) < 50:
            continue

        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunk_text = ' '.join(chunk_text.split()) 
        # For chunks after the first one, clean up partial sentences at start
        # if i > 0:
        #     # Try to start at a sentence boundary if possible
        #     sentences = chunk_text.split('. ')
        #     if len(sentences) > 1:
        #         # Join from the second sentence to avoid starting mid-sentence
        #         chunk_text = '. '.join(sentences[1:])

        if i > 0:
            # Find the first proper sentence end (". ", "! ", "? ")
            import re
            # Look for sentence endings followed by capital letter
            match = re.search(r'[.!?]\s+[A-Z]', chunk_text)
            if match:
                # Start from after the sentence end
                start_idx = match.end() - 1  # Keep the capital letter
                chunk_text = chunk_text[start_idx:]
            else:
                # No clear sentence boundary, start after first 50 chars
                if len(chunk_text) > 100:
                    # Try to find a space after position 50
                    space_idx = chunk_text.find(' ', 50)
                    if space_idx > 0:
                        chunk_text = chunk_text[space_idx+1:]
        
        # Ensure chunk isn't empty
        if not chunk_text.strip():
            continue
        
        chunks.append(chunk_text)
    
    return chunks

# def process_split_file(split_name, tokenizer):
#     """Process one split file (train, val, or test)"""
#     input_file = os.path.join(INPUT_DIR, f"{split_name}_augmented.txt")
#     output_file = os.path.join(OUTPUT_DIR, f"{split_name}_chunked.txt")
    
#     print(f"\n📄 Processing {split_name}...")
#     print(f"   Input: {input_file}")
    
#     # Read input file
#     with open(input_file, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     # Split into press releases
#     releases = split_into_press_releases(content)
#     print(f"   Found {len(releases)} press releases")
    
#     # Create chunks for each release
#     all_chunks = []
#     stats = {
#         'total_releases': len(releases),
#         'total_chunks': 0,
#         'avg_tokens_per_chunk': 0,
#         'chunks_per_release': [],
#         'token_counts': []
#     }
    
#     for i, release in enumerate(releases):
#         chunks = create_clean_chunks(release, tokenizer, chunk_id=i+1)
#         all_chunks.extend(chunks)
#         stats['chunks_per_release'].append(len(chunks))
        
#         # Count tokens in each chunk
#         for chunk in chunks:
#             tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
#             stats['token_counts'].append(tokens)
    
#     stats['total_chunks'] = len(all_chunks)
#     if stats['token_counts']:
#         stats['avg_tokens_per_chunk'] = sum(stats['token_counts']) / len(stats['token_counts'])
    
#     # Save chunks (separated by double newlines)
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write('\n\n'.join(all_chunks))
    
#     # Save statistics
#     stats_file = os.path.join(OUTPUT_DIR, f"{split_name}_stats.json")
#     with open(stats_file, 'w', encoding='utf-8') as f:
#         json.dump(stats, f, indent=2)
    
#     # Print summary
#     print(f"✅ Saved {len(all_chunks)} chunks to {output_file}")
#     print(f"   Avg tokens per chunk: {stats['avg_tokens_per_chunk']:.0f}")
#     print(f"   Min tokens: {min(stats['token_counts']) if stats['token_counts'] else 0}")
#     print(f"   Max tokens: {max(stats['token_counts']) if stats['token_counts'] else 0}")
    
#     # Show example of first chunk
#     if all_chunks:
#         print(f"\n   First chunk preview:")
#         first_lines = all_chunks[0].split('\n')[:4]
#         for line in first_lines[:4]:
#             print(f"      {line[:80]}{'...' if len(line) > 80 else ''}")
    
#     return stats

def process_split_file(split_name, tokenizer):
    """Process one split file (train, val, or test)"""
    input_file = os.path.join(INPUT_DIR, f"{split_name}_augmented.txt")
    output_file = os.path.join(OUTPUT_DIR, f"{split_name}_chunked.txt")
    
    print(f"\n📄 Processing {split_name}...")
    print(f"   Input: {input_file}")
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into press releases
    releases = split_into_press_releases(content)
    print(f"   Found {len(releases)} press releases")
    
    # Create chunks for each release
    all_chunks = []
    stats = {
        'total_releases': len(releases),
        'total_chunks': 0,
        'avg_tokens_per_chunk': 0,
        'chunks_per_release': [],
        'token_counts': []
    }
    
    for i, release in enumerate(releases):
        chunks = create_clean_chunks(release, tokenizer, chunk_id=i+1)
        all_chunks.extend(chunks)
        stats['chunks_per_release'].append(len(chunks))
        
        # Count tokens in each chunk
        for chunk in chunks:
            tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            stats['token_counts'].append(tokens)
    
    stats['total_chunks'] = len(all_chunks)
    if stats['token_counts']:
        stats['avg_tokens_per_chunk'] = sum(stats['token_counts']) / len(stats['token_counts'])
    
    # 🔥 FIXED: Save ONE clean chunk per line
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            # Ensure chunk has NO internal newlines
            if isinstance(chunk, str):
                clean_chunk = ' '.join(chunk.split())
            else:
                clean_chunk = str(chunk)
            
            # Save ONE chunk per line
            f.write(clean_chunk + '\n')
    
    # Save statistics
    stats_file = os.path.join(OUTPUT_DIR, f"{split_name}_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"✅ Saved {len(all_chunks)} chunks to {output_file}")
    print(f"   Avg tokens per chunk: {stats['avg_tokens_per_chunk']:.0f}")
    print(f"   Min tokens: {min(stats['token_counts']) if stats['token_counts'] else 0}")
    print(f"   Max tokens: {max(stats['token_counts']) if stats['token_counts'] else 0}")
    
    # Show example of first chunk (clean version)
    if all_chunks:
        print(f"\n   First chunk preview (cleaned):")
        # Get the first chunk as it will be saved
        first_chunk = all_chunks[0]
        if isinstance(first_chunk, str):
            clean_first = ' '.join(first_chunk.split())
        else:
            clean_first = str(first_chunk)
        
        # Show first 150 characters
        preview = clean_first[:150]
        print(f"      '{preview}...'")
        print(f"      Length: {len(clean_first)} chars, Tokens: {stats['token_counts'][0] if stats['token_counts'] else 'N/A'}")
    
    return stats

def main():
    """Main chunking function"""
    print("=" * 60)
    print("BMW Press Release Chunker - Clean Training Data")
    print("=" * 60)
    
    # Setup
    setup_directories()
    tokenizer = load_tokenizer()
    
    # Process all splits
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    for split in splits:
        input_file = os.path.join(INPUT_DIR, f"{split}_augmented.txt")
        if os.path.exists(input_file):
            stats = process_split_file(split, tokenizer)
            all_stats[split] = stats
        else:
            print(f"\n⚠️  Skipping {split}: {input_file} not found")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Chunking Complete - Summary")
    print("=" * 60)
    
    for split in splits:
        if split in all_stats:
            stats = all_stats[split]
            print(f"\n{split.upper()}:")
            print(f"  Press releases: {stats['total_releases']}")
            print(f"  Total chunks:   {stats['total_chunks']}")
            print(f"  Avg chunk size: {stats['avg_tokens_per_chunk']:.0f} tokens")
    
    print(f"\n📁 All files saved to: {OUTPUT_DIR}/")
    print(f"   - train_chunked.txt")
    print(f"   - val_chunked.txt")
    print(f"   - test_chunked.txt")
    print(f"   - *_stats.json (statistics)")
    
    print(f"\n✅ Chunking complete! Files are ready for fine-tuning.")

if __name__ == "__main__":
    main()