#!/usr/bin/env python3
"""
Analyze BMW Press Releases from Chunked Data
Identifies press releases by hash (#) and calculates statistics
"""

from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict

def analyze_press_releases_from_chunks(chunked_file_path):
    """
    Analyze press releases from already chunked data
    
    Args:
        chunked_file_path: Path to train_chunked.txt or similar
    """
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    print(f"🔍 Analyzing: {chunked_file_path}")
    print("=" * 60)
    
    # Read the chunked data
    with open(chunked_file_path, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f if line.strip()]
    
    print(f"Total chunks in file: {len(chunks)}")
    
    # Identify press releases by finding chunks starting with '# '
    press_releases = []
    current_pr = []
    pr_id = 0
    
    for i, chunk in enumerate(chunks):
        # Check if this chunk starts a new press release
        if chunk.startswith('# ') and '##' in chunk:
            # If we have a previous PR, save it
            if current_pr:
                press_releases.append({
                    'id': pr_id,
                    'chunks': current_pr.copy(),
                    'start_chunk': i - len(current_pr),
                    'end_chunk': i - 1
                })
                pr_id += 1
                current_pr = []
        
        # Add chunk to current PR
        current_pr.append(chunk)
        
        # Debug: Print first few chunks
        if i < 3:
            print(f"\nChunk {i} (first 100 chars):")
            print(f"  '{chunk[:100]}...'")
            print(f"  Starts with '# '? {chunk.startswith('# ')}")
            print(f"  Contains '##'? {'##' in chunk}")
    
    # Don't forget the last PR
    if current_pr:
        press_releases.append({
            'id': pr_id,
            'chunks': current_pr.copy(),
            'start_chunk': len(chunks) - len(current_pr),
            'end_chunk': len(chunks) - 1
        })
    
    print(f"\n📊 Found {len(press_releases)} press releases")
    
    # Calculate statistics for each press release
    pr_stats = []
    
    for pr in press_releases:
        # Combine all chunks for this PR
        full_text = ' '.join(pr['chunks'])
        
        # Calculate tokens
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        token_count = len(tokens)
        
        # Get title (first line starting with '# ')
        title = ""
        for chunk in pr['chunks']:
            if chunk.startswith('# '):
                # Extract title (text between '# ' and '##' or end)
                title_part = chunk[2:]  # Remove '# '
                if '##' in title_part:
                    title_part = title_part.split('##')[0].strip()
                title = title_part[:100]  # First 100 chars
                break
        
        pr_stats.append({
            'id': pr['id'],
            'title': title,
            'num_chunks': len(pr['chunks']),
            'tokens': token_count,
            'words': len(full_text.split()),
            'chars': len(full_text),
            'start_chunk': pr['start_chunk'],
            'end_chunk': pr['end_chunk']
        })
    
    # Overall statistics
    if pr_stats:
        tokens = [pr['tokens'] for pr in pr_stats]
        chunks_per_pr = [pr['num_chunks'] for pr in pr_stats]
        
        print(f"\n📈 PRESS RELEASE STATISTICS:")
        print(f"  Average length: {np.mean(tokens):.0f} tokens")
        print(f"  Minimum length: {np.min(tokens)} tokens")
        print(f"  Maximum length: {np.max(tokens)} tokens")
        print(f"  Std deviation:  {np.std(tokens):.0f} tokens")
        
        print(f"\n  Average chunks per PR: {np.mean(chunks_per_pr):.1f}")
        print(f"  Min chunks per PR: {np.min(chunks_per_pr)}")
        print(f"  Max chunks per PR: {np.max(chunks_per_pr)}")
        
        print(f"\n  Total tokens in all PRs: {sum(tokens):,}")
        print(f"  Total words in all PRs: {sum(pr['words'] for pr in pr_stats):,}")
        
        # Distribution analysis
        print(f"\n📊 DISTRIBUTION BY LENGTH:")
        bins = [(0, 100), (100, 250), (250, 500), (500, 750), 
                (750, 1000), (1000, 1500), (1500, 2000), 
                (2000, 3000), (3000, 5000), (5000, 10000)]
        
        for bin_min, bin_max in bins:
            count = sum(1 for t in tokens if bin_min <= t < bin_max)
            if count > 0:
                percentage = count / len(tokens) * 100
                print(f"    {bin_min:5d}-{bin_max:5d} tokens: {count:3d} PRs ({percentage:.1f}%)")
        
        # Show examples
        print(f"\n🎯 SAMPLE PRESS RELEASES:")
        
        # Shortest PR
        shortest = min(pr_stats, key=lambda x: x['tokens'])
        print(f"  Shortest PR (ID {shortest['id']}):")
        print(f"    Title: {shortest['title']}")
        print(f"    Length: {shortest['tokens']} tokens, {shortest['num_chunks']} chunks")
        
        # Longest PR  
        longest = max(pr_stats, key=lambda x: x['tokens'])
        print(f"\n  Longest PR (ID {longest['id']}):")
        print(f"    Title: {longest['title']}")
        print(f"    Length: {longest['tokens']} tokens, {longest['num_chunks']} chunks")
        
        # Most chunked PR
        most_chunks = max(pr_stats, key=lambda x: x['num_chunks'])
        print(f"\n  Most chunked PR (ID {most_chunks['id']}):")
        print(f"    Title: {most_chunks['title']}")
        print(f"    Chunks: {most_chunks['num_chunks']}, Tokens: {most_chunks['tokens']}")
        print(f"    Avg tokens per chunk: {most_chunks['tokens'] / most_chunks['num_chunks']:.0f}")
        
        # Check if 512 tokens is optimal
        print(f"\n✅ OPTIMAL CHUNK SIZE ANALYSIS:")
        chunks_needed_512 = sum(np.ceil(t / 512) for t in tokens)
        chunks_needed_384 = sum(np.ceil(t / 384) for t in tokens)
        chunks_needed_768 = sum(np.ceil(t / 768) for t in tokens)
        
        print(f"  With 512-token chunks: {chunks_needed_512:.0f} chunks needed")
        print(f"  With 384-token chunks: {chunks_needed_384:.0f} chunks needed")
        print(f"  With 768-token chunks: {chunks_needed_768:.0f} chunks needed")
        
        actual_chunks = len(chunks)
        print(f"\n  Actual chunks created: {actual_chunks}")
        print(f"  Efficiency vs 512-token: {chunks_needed_512 / actual_chunks * 100:.1f}%")
        
        # Save detailed stats
        import json
        output_file = chunked_file_path.replace('.txt', '_pr_stats.json')
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_press_releases': len(press_releases),
                    'total_chunks': len(chunks),
                    'avg_tokens_per_pr': float(np.mean(tokens)),
                    'min_tokens_per_pr': int(np.min(tokens)),
                    'max_tokens_per_pr': int(np.max(tokens)),
                    'avg_chunks_per_pr': float(np.mean(chunks_per_pr)),
                    'chunks_needed_512': float(chunks_needed_512),
                    'chunks_needed_384': float(chunks_needed_384),
                    'chunks_needed_768': float(chunks_needed_768),
                    'efficiency_vs_512': float(chunks_needed_512 / actual_chunks * 100)
                },
                'press_releases': pr_stats
            }, f, indent=2)
        
        print(f"\n📁 Detailed statistics saved to: {output_file}")
    
    tokens = [pr['tokens'] for pr in pr_stats]

    print(f"\n📊 PERCENTILE ANALYSIS:")
    print(f"  25th percentile: {np.percentile(tokens, 25):.0f} tokens")
    print(f"  50th percentile (MEDIAN): {np.percentile(tokens, 50):.0f} tokens") 
    print(f"  75th percentile: {np.percentile(tokens, 75):.0f} tokens")
    print(f"  90th percentile: {np.percentile(tokens, 90):.0f} tokens")
    print(f"  95th percentile: {np.percentile(tokens, 95):.0f} tokens")

    # Also show how many chunks each percentile needs
    print(f"\n📦 CHUNKS NEEDED AT 512 TOKENS:")
    for p in [25, 50, 75, 90, 95]:
        tokens_at_p = np.percentile(tokens, p)
        chunks_needed = np.ceil(tokens_at_p / 512)
        print(f"  {p}th percentile ({tokens_at_p:.0f}t): {chunks_needed:.0f} chunks")


    return press_releases, pr_stats

# Also create a simpler version for quick analysis
def quick_analyze_chunked_file(chunked_file_path):
    """Quick analysis showing key metrics"""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    with open(chunked_file_path, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f if line.strip()]
    
    print(f"\n🚀 QUICK ANALYSIS: {chunked_file_path}")
    print("=" * 50)
    
    # Count press releases (chunks starting with '# ')
    pr_starts = [i for i, chunk in enumerate(chunks) 
                if chunk.startswith('# ') and '##' in chunk]
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Press releases found: {len(pr_starts)}")
    
    if pr_starts:
        # Calculate average PR length in chunks
        chunk_counts = []
        for i in range(len(pr_starts)):
            if i < len(pr_starts) - 1:
                chunk_counts.append(pr_starts[i+1] - pr_starts[i])
            else:
                chunk_counts.append(len(chunks) - pr_starts[i])
        
        print(f"Avg chunks per PR: {np.mean(chunk_counts):.1f}")
        print(f"Min chunks per PR: {np.min(chunk_counts)}")
        print(f"Max chunks per PR: {np.max(chunk_counts)}")
        
        # Analyze chunk sizes
        chunk_tokens = []
        for chunk in chunks[:100]:  # Sample first 100
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            chunk_tokens.append(len(tokens))
        
        print(f"\nChunk size (sample of 100):")
        print(f"  Avg: {np.mean(chunk_tokens):.0f} tokens")
        print(f"  Min: {np.min(chunk_tokens)} tokens")
        print(f"  Max: {np.max(chunk_tokens)} tokens")
        print(f"  Target: 512 tokens")
        
        # How many are close to target
        good_chunks = sum(1 for t in chunk_tokens if 450 <= t <= 550)
        print(f"  {good_chunks}/100 are 450-550 tokens ({good_chunks}%)")

        
if __name__ == "__main__":
    print("=" * 60)
    print("BMW PRESS RELEASE ANALYZER")
    print("=" * 60)
    
    # File to analyze
    chunked_file = "bmw_press_datasets/chunked/train_chunked.txt"
    
    # Quick analysis
    quick_analyze_chunked_file(chunked_file)
    
    # Full analysis
    print("\n" + "=" * 60)
    print("FULL ANALYSIS")
    print("=" * 60)
    
    press_releases, stats = analyze_press_releases_from_chunks(chunked_file)
    
    # Also analyze val and test
    for split in ['val', 'test']:
        file_path = f"bmw_press_datasets/chunked/{split}_chunked.txt"
        try:
            print(f"\n\nAnalyzing {split} data...")
            quick_analyze_chunked_file(file_path)
        except FileNotFoundError:
            print(f"⚠️  {file_path} not found")