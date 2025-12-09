#!/usr/bin/env python3
"""
Comprehensive PII audit for BMW dataset - Shows ALL matches
"""

import re
import os

def audit_pii_detailed(file_path, output_report=None):
    """Check file for potential PII and print ALL matches"""
    
    pii_patterns = {
        'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phones': r'(?:\+\d{1,4}[-\s]?)?\(?\d{1,6}\)?[-\s.]?\d{1,6}[-\s.]?\d{1,6}[-\s.]?\d{0,6}',
        'ibans': r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b',
        'urls': r'https?://[^\s<>"\'{}|\\^`\[\]]+',
    }
    
    print(f"\n{'='*60}")
    print(f"🔍 DETAILED PII AUDIT: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    all_matches = {}
    
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        results[pii_type] = len(matches)
        all_matches[pii_type] = matches
        
        # Print summary
        if matches:
            print(f"\n📊 {pii_type.upper()}: {len(matches)} found")
            
            # Show first 10 samples
            print(f"   First 10 samples:")
            for i, match in enumerate(matches[:10]):
                print(f"     {i+1:2d}. {match}")
            
            if len(matches) > 10:
                print(f"     ... and {len(matches) - 10} more")
            
            # Show unique patterns (for emails - domains, for phones - formats)
            if pii_type == 'emails':
                domains = {}
                for email in matches:
                    domain = email.split('@')[1] if '@' in email else 'unknown'
                    domains[domain] = domains.get(domain, 0) + 1
                
                print(f"\n   📈 Email Domain Breakdown:")
                for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                    print(f"     {domain}: {count}")
            
            elif pii_type == 'phones':
                # Group by country code pattern
                country_codes = {}
                for phone in matches:
                    # Extract country code if present
                    if phone.startswith('+'):
                        country_code = re.match(r'\+\d{1,4}', phone)
                        if country_code:
                            code = country_code.group(0)
                            country_codes[code] = country_codes.get(code, 0) + 1
                        else:
                            country_codes['unknown'] = country_codes.get('unknown', 0) + 1
                    else:
                        country_codes['no_country_code'] = country_codes.get('no_country_code', 0) + 1
                
                if country_codes:
                    print(f"\n   📈 Phone Country Code Breakdown:")
                    for code, count in sorted(country_codes.items(), key=lambda x: x[1], reverse=True):
                        print(f"     {code}: {count}")
        
        else:
            print(f"\n✅ {pii_type.upper()}: 0 found")
    
    # Print a summary table
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    for pii_type, count in results.items():
        status = "❌" if count > 0 else "✅"
        print(f"  {status} {pii_type.upper():10s}: {count:4d}")
    
    # Export all matches to a file if requested
    if output_report:
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(f"PII Audit Report for: {file_path}\n")
            f.write(f"Generated: {os.path.getmtime(file_path)}\n")
            f.write("="*60 + "\n\n")
            
            for pii_type, matches in all_matches.items():
                if matches:
                    f.write(f"{pii_type.upper()} ({len(matches)} found):\n")
                    f.write("-"*40 + "\n")
                    for match in matches:
                        f.write(f"  {match}\n")
                    f.write("\n")
        
        print(f"\n📄 Full report saved to: {output_report}")
    
    return results, all_matches

def find_lines_with_pii(file_path, pii_type, pattern):
    """Find which lines contain specific PII types"""
    
    print(f"\n🔍 Finding lines with {pii_type} in {file_path}")
    print("-"*60)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    matches_found = 0
    for line_num, line in enumerate(lines, 1):
        matches = re.findall(pattern, line, re.IGNORECASE)
        if matches:
            matches_found += 1
            print(f"\nLine {line_num}:")
            print(f"  Content: {line.strip()[:100]}...")
            print(f"  Matches: {matches}")
    
    if matches_found == 0:
        print(f"  No {pii_type} found in any line")
    
    return matches_found

# Main execution
if __name__ == "__main__":
    print("🚀 BMW PRESS RELEASE - DETAILED PII AUDIT")
    print("="*60)
    
    # Files to audit (modify as needed)
    files_to_audit = [
        "bmw_press_datasets/train_balanced_cleaned.txt",
        "bmw_press_datasets/val_balanced_cleaned.txt", 
        "bmw_press_datasets/test_balanced_cleaned.txt"
    ]
    
    # Optionally audit original files for comparison
    compare_with_originals = True
    
    all_results = {}
    
    for file_path in files_to_audit:
        if os.path.exists(file_path):
            # Generate report filename
            base_name = os.path.basename(file_path).replace('.txt', '')
            report_file = f"pii_report_{base_name}.txt"
            
            # Run detailed audit
            results, matches = audit_pii_detailed(file_path, report_file)
            all_results[file_path] = results
            
            # Special debug for emails - find which lines they're on
            if results and results.get('emails', 0) > 0:
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                find_lines_with_pii(file_path, 'emails', email_pattern)
    
    # Compare with originals if requested
    if compare_with_originals:
        print(f"\n{'='*60}")
        print("🔄 COMPARISON WITH ORIGINAL FILES")
        print(f"{'='*60}")
        
        for file_path in files_to_audit:
            original_path = file_path.replace('_balanced_cleaned', '')
            if os.path.exists(original_path):
                print(f"\nComparing: {os.path.basename(original_path)} → {os.path.basename(file_path)}")
                
                with open(original_path, 'r', encoding='utf-8') as f:
                    orig_content = f.read()
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    clean_content = f.read()
                
                # Count emails in both
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                orig_emails = re.findall(email_pattern, orig_content, re.IGNORECASE)
                clean_emails = re.findall(email_pattern, clean_content, re.IGNORECASE)
                
                print(f"  Emails removed: {len(orig_emails)} → {len(clean_emails)} "
                      f"({len(orig_emails) - len(clean_emails)} removed, "
                      f"{100*(len(orig_emails)-len(clean_emails))/max(1, len(orig_emails)):.1f}%)")
                
                # Show which emails remain
                if clean_emails:
                    remaining_domains = set(email.split('@')[1] for email in clean_emails if '@' in email)
                    print(f"  Remaining domains: {', '.join(sorted(remaining_domains))}")