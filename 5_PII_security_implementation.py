
"""
COMPREHENSIVE PII CLEANUP WITH BALANCED URL STRATEGY
Balanced approach: Keep URLs but sanitize with context preservation
"""

import re
import os
from pathlib import Path

def balanced_url_sanitization(content):
    """
    Balanced URL strategy - Keep context but remove exact links
    Based on your url_sanitizer.py approach
    """
    
    # Track changes for reporting
    url_changes = {
        'corporate_sites': 0,
        'social_media': 0,
        'external_links': 0
    }
    
    # 1. CORPORATE DOMAINS - Keep brand context, remove exact paths
    corporate_patterns = [
        # BMW Group official sites
        (r'https?://(?:www\.)?bmwgroup\.(com|de)/([^\s]*)', 
         lambda m: f"Official BMW Group website ({m.group(2).split('/')[0] if '/' in m.group(2) else 'homepage'})"),
        
        # BMW brand sites
        (r'https?://(?:www\.)?bmw\.(com|de|co\.uk)/([^\s]*)',
         lambda m: f"BMW official website ({m.group(2).split('/')[0] if '/' in m.group(2) else 'homepage'})"),
        
        # MINI brand sites
        (r'https?://(?:www\.)?mini\.(com|co\.uk)/([^\s]*)',
         lambda m: f"MINI official website ({m.group(2).split('/')[0] if '/' in m.group(2) else 'homepage'})"),
        
        # Rolls-Royce Motor Cars
        (r'https?://(?:www\.)?(press\.)?rolls-roycemotorcars?\.(com|co\.uk)/([^\s]*)',
         lambda m: f"Rolls-Royce Motor Cars official website"),
    ]
    
    for pattern, replacement in corporate_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            url_changes['corporate_sites'] += len(matches)
            if callable(replacement):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            else:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    # 2. SOCIAL MEDIA - Keep platform reference with brand context
    social_platforms = {
        'linkedin': 'LinkedIn',
        'youtube': 'YouTube', 
        'instagram': 'Instagram',
        'twitter|x': 'Twitter/X',
        'facebook': 'Facebook',
        'tiktok': 'TikTok',
        'pinterest': 'Pinterest'
    }
    
    for pattern, name in social_platforms.items():
        # Match social media URLs with brand references
        social_pattern = f'https?://(?:www\.)?{pattern}\.com/([^\\s]*)'
        matches = re.findall(social_pattern, content, re.IGNORECASE)
        
        if matches:
            url_changes['social_media'] += len(matches)
            
            # Determine brand from URL path when possible
            def social_replacement(match):
                path = match.group(1).lower() if match.group(1) else ''
                brand = "BMW Group"
                
                # Extract brand from path if identifiable
                if 'bmwmotorrad' in path or 'motorrad' in path:
                    brand = "BMW Motorrad"
                elif 'mini' in path:
                    brand = "MINI"
                elif 'rollsroyce' in path or 'rolls-royce' in path:
                    brand = "Rolls-Royce"
                
                return f"{brand} on {name}"
            
            content = re.sub(social_pattern, social_replacement, content, flags=re.IGNORECASE)
    
    # 3. EXTERNAL/OTHER URLs - Generic replacement
    other_urls = re.findall(r'https?://[^\s<>"\']+', content)
    if other_urls:
        url_changes['external_links'] = len(other_urls)
        # Replace with generic placeholder that maintains some context
        def external_replacement(match):
            return "[EXTERNAL_RESOURCE]"
        
        content = re.sub(r'https?://[^\s<>"\']+', external_replacement, content)
    
    return content, url_changes

def comprehensive_pii_clean_balanced(file_path, output_path):
    """One-pass cleanup with balanced URL strategy"""
    
    print(f"\n🔧 Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_length = len(content)
    changes = {}
    
    # 1. COMPLETE EMAIL REMOVAL - All domains found in your audit
    print("  Removing emails...")
    email_patterns = [
        # Country-specific BMW domains from your audit
        r'\b[A-Za-z0-9._%+-]+@bmw\.(co\.uk|co\.za|com|de)\b',
        r'\b[A-Za-z0-9._%+-]+@bmwgroup\.(ca|com|de)\b',
        r'\b[A-Za-z0-9._%+-]+@mini\.(co\.uk|com)\b',
        # Partner/other domains from audit
        r'\b[A-Za-z0-9._%+-]+@(haebmau\.de|esmt\.org|broadarrowauctions\.com|bmwmc\.com)\b',
        # Rolls-Royce domains (as discussed)
        r'\b[A-Za-z0-9._%+-]+@rolls-royce(motorcars)?\.(com|co\.uk)\b',
        # Catch-all for any remaining emails (safety net)
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    ]
    
    total_emails_removed = 0
    for pattern in email_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            total_emails_removed += len(matches)
            content = re.sub(pattern, '[CONTACT_EMAIL]', content, flags=re.IGNORECASE)
    
    if total_emails_removed > 0:
        changes['emails_removed'] = total_emails_removed
    
    # 2. PRECISE PHONE NUMBER REMOVAL (only actual phone patterns)
    print("  Removing phone numbers...")
    phone_patterns = [
        # German phone formats (most common in your data)
        r'\+\d{1,4}[-\s]?\(?\d{1,6}\)?[-\s.]?\d{1,6}[-\s.]?\d{1,6}',
        # UK mobile formats (07815 xxx xxx from your data)
        r'\b07\d{3}[-\s.]?\d{3}[-\s.]?\d{3}\b',
        r'\b0[1-8]\d{2,3}[-\s.]?\d{3,4}[-\s.]?\d{3,4}\b',
        # North American formats (905-xxx-xxxx from your data)
        r'\b[2-9]\d{2}[-\s.]?\d{3}[-\s.]?\d{4}\b',
        # International with country code (common patterns)
        r'\+\d{1,4}[-\s]?\d{6,12}\b',
        # Phone numbers with extensions
        r'\b(?:\+\d{1,4}[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}(?:[-\s.]?x\.?\d{1,5})?\b',
    ]
    
    total_phones_removed = 0
    for pattern in phone_patterns:
        matches = re.findall(pattern, content)
        if matches:
            total_phones_removed += len(matches)
            content = re.sub(pattern, '[CONTACT_PHONE]', content)
    
    if total_phones_removed > 0:
        changes['phones_removed'] = total_phones_removed
    
    # 3. BALANCED URL SANITIZATION (Your preferred approach)
    print("  Sanitizing URLs with balanced strategy...")
    content, url_changes = balanced_url_sanitization(content)
    
    # Update changes with URL statistics
    for url_type, count in url_changes.items():
        if count > 0:
            changes[f'urls_{url_type}'] = count
    
    # 4. OPTIONAL: Name pseudonymization (less aggressive)
    print("  Applying light name pseudonymization...")
    # Only target names when they appear in specific contact patterns
    name_context_patterns = [
        # Names followed by contact titles
        (r'\b(Chris|Christina|Tom|Samuel|Helen|Marc|Barb|Jean-Fran[çc]ois|Angela|Steve)\.?\s+[A-Z][a-z]+\b(?=\s*(?:Press|Media|Communications|Officer|Manager|Director))',
         '[MEDIA_REPRESENTATIVE]'),
        # Common executive names
        (r'\b(Oliver\s+Zipse|Nicolas\s+Peter|Jochen\s+Goller|Ilka\s+Horstmeier)\b',
         '[EXECUTIVE_OFFICER]'),
    ]
    
    for pattern, replacement in name_context_patterns:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    # 5. ADDITIONAL PRIVACY: Remove specific address patterns
    address_patterns = [
        # German address patterns
        (r'\b\d{5}\s+[A-Z][a-züäöß]+\b', '[LOCATION]'),
        # Street address patterns
        (r'\b[A-Z][a-z]+\s+\d{1,3}[a-z]?\b', '[ADDRESS]'),
    ]
    
    for pattern, replacement in address_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write cleaned file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Calculate compression/change percentage
    final_length = len(content)
    change_percentage = 100 * (original_length - final_length) / original_length if original_length > 0 else 0
    
    # Verification and reporting
    print(f"✅ Saved to: {output_path}")
    print(f"📊 Statistics:")
    print(f"   - Original size: {original_length:,} chars")
    print(f"   - Final size: {final_length:,} chars")
    print(f"   - Data reduction: {change_percentage:.1f}%")
    
    if changes:
        print(f"📋 Changes made:")
        for change_type, count in changes.items():
            readable_type = change_type.replace('_', ' ').title()
            print(f"   - {readable_type}: {count:,}")
    
    # Show sample of cleaned content
    print(f"\n📄 Sample of cleaned content (first 2 lines):")
    lines = content.split('\n')
    for i, line in enumerate(lines[:2]):
        if line.strip():
            preview = line[:120] + '...' if len(line) > 120 else line
            print(f"   {i+1}. {preview}")
    
    return changes

def run_balanced_pipeline():
    """Run complete PII cleanup with balanced URL strategy"""
    
    print("="*70)
    print("🚀 BMW PRESS RELEASE - BALANCED PII CLEANUP PIPELINE")
    print("="*70)
    print("Strategy: Remove ALL PII, keep URL context with balanced sanitization")
    print("-"*70)
    
    splits = ['train', 'val', 'test']
    all_changes = {}
    
    for split in splits:
        input_file = f"bmw_press_datasets/{split}.txt"
        output_file = f"bmw_press_datasets/{split}_balanced_cleaned.txt"
        
        if os.path.exists(input_file):
            print(f"\n{'='*40}")
            print(f"📁 Processing: {split.upper()}")
            changes = comprehensive_pii_clean_balanced(input_file, output_file)
            all_changes[split] = changes
            
            # Quick verification audit
            print(f"\n🔍 Quick verification audit:")
            with open(output_file, 'r', encoding='utf-8') as f:
                cleaned_content = f.read()
            
            # Check for remaining PII
            remaining_emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', cleaned_content, re.IGNORECASE))
            remaining_urls = len(re.findall(r'https?://[^\s<>"\']+', cleaned_content))
            remaining_phones = len(re.findall(r'\+\d{1,4}[-\s]?\d{1,14}(?:[-\s]\d{1,13})?', cleaned_content))
            
            print(f"   ✅ Remaining emails: {remaining_emails}")
            print(f"   ✅ Remaining raw URLs: {remaining_urls}")
            print(f"   ✅ Remaining phone patterns: {remaining_phones}")
            
            if remaining_emails == 0 and remaining_urls == 0:
                print(f"   🎉 All PII successfully removed!")
            else:
                print(f"   ⚠️  Some PII may remain. Check the detailed audit.")
            
        else:
            print(f"❌ Input file not found: {input_file}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("📋 PIPELINE COMPLETION SUMMARY")
    print(f"{'='*70}")
    
    total_changes = {}
    for split in splits:
        if split in all_changes:
            print(f"\n{split.upper()}:")
            for change_type, count in all_changes[split].items():
                readable_type = change_type.replace('_', ' ').title()
                print(f"  {readable_type}: {count:,}")
                
                # Aggregate totals
                if change_type in total_changes:
                    total_changes[change_type] += count
                else:
                    total_changes[change_type] = count
    
    print(f"\n{'='*70}")
    print("📈 TOTALS ACROSS ALL DATASETS")
    print(f"{'='*70}")
    
    for change_type, total in total_changes.items():
        readable_type = change_type.replace('_', ' ').title()
        print(f"  {readable_type}: {total:,}")
    
    print(f"\n✅ All cleaned files saved as: *_balanced_cleaned.txt")
    print(f"📁 Files ready for training:")
    for split in splits:
        print(f"   - bmw_press_datasets/{split}_balanced_cleaned.txt")
    
    print(f"\n⚠️  NEXT STEP: Run your PII audit on the cleaned files to verify:")
    print(f"    python dirty_pii_check.py  # Update to check *_balanced_cleaned.txt")

def verify_with_original_audit():
    """Quick function to verify the cleaned files match your audit expectations"""
    
    print(f"\n{'='*70}")
    print("🔍 VERIFICATION: Running PII audit on cleaned files")
    print(f"{'='*70}")
    
    # Simple verification function
    def quick_verify(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', content, re.IGNORECASE)
        urls = re.findall(r'https?://[^\s<>"\']+', content)
        phones = re.findall(r'\+\d{1,4}[-\s]?\d{1,14}(?:[-\s]\d{1,13})?', content)
        
        return len(emails), len(urls), len(phones)
    
    for split in ['train', 'val', 'test']:
        cleaned_file = f"bmw_press_datasets/{split}_balanced_cleaned.txt"
        if os.path.exists(cleaned_file):
            emails, urls, phones = quick_verify(cleaned_file)
            print(f"\n{split.upper()}_balanced_cleaned.txt:")
            print(f"  Emails: {emails} (expecting 0)")
            print(f"  Raw URLs: {urls} (expecting 0 - replaced with contextual text)")
            print(f"  Phone patterns: {phones} (expecting <50, not 12,000+)")
            
            if emails == 0 and urls == 0:
                print(f"  ✅ PASS: No direct PII remains")
            else:
                print(f"  ⚠️  Check needed")

if __name__ == "__main__":
    # Run the balanced cleanup pipeline
    run_balanced_pipeline()
    
    # Optional: Run verification
    verify_with_original_audit()
    
    print(f"\n🎯 Your dataset is now ready for fine-tuning!")
    print(f"   The balanced approach preserves URL context while removing direct PII.")
    print(f"   This helps your model learn about information sources without privacy risks.")
