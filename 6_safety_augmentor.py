
"""
STANDALONE SAFETY AUGMENTOR
Run this AFTER PII cleanup to add safety Q&A patterns
"""

import os
import random

def add_safety_patterns(content, strategy="append"):
    """
    Add safe Q&A patterns to teach model proper behavior
    
    Args:
        content: Text content to augment
        strategy: "append" (add at end), "prepend" (add at beginning), 
                 "mix" (randomly insert throughout)
    
    Returns:
        Augmented content and count of patterns added
    """
    
    safety_qa = [
        # Contact information redirection
        ("How can I contact BMW?", 
         "For official contact, please visit the BMW Group website and use the contact form. "
         "This ensures your inquiry reaches the correct department."),
        
        ("What is BMW's email address?", 
         "BMW Group handles inquiries through their official website contact form. "
         "Visit their website for the most current and secure contact methods."),
        
        ("Phone number for BMW headquarters?", 
         "The main switchboard can be reached during business hours. "
         "For specific department contacts, please check the official website."),
        
        ("Where is BMW located?", 
         "BMW Group headquarters are in Munich, Germany. "
         "For specific location information, please refer to their official website."),
        
        ("Financial report information?", 
         "All financial reports and investor information are available in the "
         "Investor Relations section of BMW Group's official website."),
        
        ("Who is the CEO of BMW?", 
         "The Board of Management leads BMW Group. "
         "Executive leadership details are available on the official corporate website."),
        
        ("Press contact details for BMW?", 
         "Press inquiries should be directed through the BMW Group Press Club portal. "
         "Visit their official website for media resources and contacts."),
        
        ("How to invest in BMW?", 
         "Investment information is available through BMW Group's Investor Relations. "
         "Always consult official sources for financial decisions."),
        
        ("BMW's social media accounts?", 
         "BMW Group maintains official social media channels. "
         "Please visit their website for verified social media links."),
        
        ("Where to find BMW product specifications?", 
         "Detailed product information and specifications are available on the official BMW website. "
         "Always refer to official sources for accurate technical data."),
        
        ("How to get BMW service information?", 
         "Service and maintenance information is available through authorized BMW dealers "
         "and the official BMW Group website."),
        
        ("BMW corporate sustainability report?", 
         "Sustainability reports and corporate responsibility information are published "
         "in the Sustainability section of BMW Group's official website."),
        
        ("Where to download BMW annual report?", 
         "Annual reports and financial publications are available in the "
         "Investor Relations section of the official BMW Group website."),
        
        ("How to contact BMW customer service?", 
         "Customer service inquiries should be directed through the official contact channels "
         "on the BMW Group website for prompt and accurate assistance."),
    ]
    
    # Convert to training format - multiple formats for better learning
    augmented_patterns = []
    for question, answer in safety_qa:
        # Format 1: Q/A style (most explicit)
        augmented_patterns.append(f"Q: {question}\nA: {answer}")
        
        # Format 2: Conversation style
        augmented_patterns.append(f"User: {question}\nAssistant: {answer}")
        
        # Format 3: Instruction style
        augmented_patterns.append(f"Instruction: Answer this question about BMW.\nQuestion: {question}\nResponse: {answer}")
        
        # Format 4: Single line (for variety)
        augmented_patterns.append(f"{question} {answer}")
    
    patterns_added = len(augmented_patterns)
    
    # Apply augmentation strategy
    if strategy == "append":
        # Add all patterns at the end
        content += "\n" + "\n".join(augmented_patterns)
        
    elif strategy == "prepend":
        # Add all patterns at the beginning
        content = "\n".join(augmented_patterns) + "\n" + content
        
    elif strategy == "mix":
        # Randomly insert patterns throughout
        lines = content.split('\n')
        
        for pattern in augmented_patterns:
            # Insert at random position (avoid very beginning/end)
            insert_pos = random.randint(len(lines) // 10, len(lines) - len(lines) // 10)
            lines.insert(insert_pos, pattern)
        
        content = '\n'.join(lines)
    
    elif strategy == "weighted":
        # Add patterns proportionally to dataset size
        lines = content.split('\n')
        target_augmentation = min(len(lines) // 50, 100)  # ~2% of dataset
        
        # Shuffle and add limited number
        random.shuffle(augmented_patterns)
        patterns_to_add = augmented_patterns[:target_augmentation]
        patterns_added = len(patterns_to_add)
        
        for pattern in patterns_to_add:
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, pattern)
        
        content = '\n'.join(lines)
    
    return content, patterns_added

def process_dataset_split(input_path, output_path, strategy="append"):
    """
    Process a single dataset split with safety augmentation
    """
    print(f"\n📁 Processing: {os.path.basename(input_path)}")
    
    # Read cleaned file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_lines = len(content.split('\n'))
    print(f"   Original: {original_lines:,} lines")
    
    # Apply safety augmentation
    augmented_content, patterns_added = add_safety_patterns(content, strategy)
    
    final_lines = len(augmented_content.split('\n'))
    print(f"   Added: {patterns_added:,} safety patterns")
    print(f"   Final: {final_lines:,} lines")
    print(f"   Augmentation: +{((final_lines - original_lines) / original_lines * 100):.1f}%")
    
    # Save augmented file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(augmented_content)
    
    print(f"   ✅ Saved to: {output_path}")
    
    # Show sample of added patterns
    print(f"\n   📄 Sample added patterns:")
    augmented_lines = augmented_content.split('\n')
    for i, line in enumerate(augmented_lines[-6:]):  # Last 6 lines
        if line.strip():
            preview = line[:80] + '...' if len(line) > 80 else line
            print(f"      {preview}")
    
    return patterns_added

def main():
    """Main function to run safety augmentation on all splits"""
    
    print("="*70)
    print("🚀 BMW PRESS RELEASE - SAFETY AUGMENTATION PIPELINE")
    print("="*70)
    print("Adding safe Q&A patterns to teach model proper PII handling")
    print("-"*70)
    
    # Configuration
    input_dir = "bmw_press_datasets/cleaned"
    output_dir = "bmw_press_datasets/augmented"
    augmentation_strategy = "weighted"  # Options: "append", "prepend", "mix", "weighted"
    
    splits = ['train', 'val', 'test']
    total_patterns = 0
    
    for split in splits:
        input_file = os.path.join(input_dir, f"{split}_balanced_cleaned.txt")
        output_file = os.path.join(output_dir, f"{split}_augmented.txt")
        
        if os.path.exists(input_file):
            patterns = process_dataset_split(input_file, output_file, augmentation_strategy)
            total_patterns += patterns
        else:
            print(f"\n⚠️  Input file not found: {input_file}")
            print(f"   Looking for: {os.path.abspath(input_file)}")
            print(f"   Available files in {input_dir}:")
            if os.path.exists(input_dir):
                for f in os.listdir(input_dir):
                    print(f"     - {f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 AUGMENTATION SUMMARY")
    print(f"{'='*70}")
    print(f"Strategy used: {augmentation_strategy}")
    print(f"Total safety patterns added: {total_patterns:,}")
    print(f"\n📁 Augmented files saved to: {output_dir}/")
    for split in splits:
        print(f"   - {split}_augmented.txt")
    
    print(f"\n🎯 These files are now READY for training!")
    print(f"   The model will learn safe responses to PII-related questions.")
    
    # Verify file sizes
    print(f"\n🔍 File size verification:")
    for split in splits:
        input_file = os.path.join(input_dir, f"{split}_balanced_cleaned.txt")
        output_file = os.path.join(output_dir, f"{split}_augmented.txt")
        
        if os.path.exists(input_file) and os.path.exists(output_file):
            with open(input_file, 'r') as f:
                input_lines = len(f.readlines())
            with open(output_file, 'r') as f:
                output_lines = len(f.readlines())
            
            increase_pct = ((output_lines - input_lines) / input_lines * 100) if input_lines > 0 else 0
            print(f"   {split}: {input_lines:,} → {output_lines:,} lines (+{increase_pct:.1f}%)")

if __name__ == "__main__":
    main()