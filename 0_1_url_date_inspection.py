import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json

class BMWDateTester:
    def __init__(self, url):
        self.url = url
        self.selectors = [
            # Your original selector
            "#content > div > div.content-left > div > div.article-info > h5 > span.date",
            # Alternative selectors
            "span.date",
            ".date",
            ".article-date",
            ".press-date",
            ".release-date",
            ".published-date",
            "time",
            "meta[property='article:published_time']",
            "meta[name='date']",
            "meta[name='publication_date']",
            "meta[name='publish_date']",
            "meta[itemprop='datePublished']",
            # BMW specific classes
            ".press-release-date",
            ".pressdetail-date",
            ".publication-date",
            # Header elements that might contain dates
            "h5",
            ".date-info",
            ".article-header .date"
        ]
    
    def clean_text(self, element):
        """Clean text from HTML element"""
        if element is None:
            return None
        
        if hasattr(element, 'get_text'):
            text = element.get_text(separator=" ", strip=True)
        elif hasattr(element, 'get'):
            text = element.get('content', '') or element.get('datetime', '')
        else:
            text = str(element)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else None
    
    def parse_date(self, date_text):
        """Try to parse date from various formats"""
        if not date_text:
            return None, None
        
        date_patterns = [
            # European format (DD.MM.YYYY)
            r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
            # ISO format (YYYY-MM-DD)
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            # US format (MM/DD/YYYY)
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            # With month names
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})'
        ]
        
        month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if len(groups) == 3:
                    # DD.MM.YYYY format
                    if '.' in date_text[match.start():match.end()]:
                        day, month, year = groups
                        return datetime(int(year), int(month), int(day)), int(year)
                    # YYYY-MM-DD format
                    elif '-' in date_text[match.start():match.end()]:
                        year, month, day = groups
                        return datetime(int(year), int(month), int(day)), int(year)
                    # MM/DD/YYYY format
                    elif '/' in date_text[match.start():match.end()]:
                        month, day, year = groups
                        return datetime(int(year), int(month), int(day)), int(year)
                
                # Month name format
                elif len(groups) == 3:
                    if any(month in groups[1].lower() for month in month_names.keys()):
                        for i, group in enumerate(groups):
                            if group.lower() in month_names:
                                month_num = month_names[group.lower()]
                                # Check surrounding groups for day and year
                                if i == 0:  # Format: DD Month YYYY
                                    day = int(groups[0])
                                    year = int(groups[2])
                                else:  # Format: Month DD, YYYY
                                    day = int(groups[1])
                                    year = int(groups[2])
                                return datetime(year, month_num, day), year
        
        return None, None
    
    def test_selectors(self, soup):
        """Test all selectors on the page"""
        results = []
        
        print("Testing all selectors:")
        print("-" * 50)
        
        for i, selector in enumerate(self.selectors, 1):
            try:
                if selector.startswith("meta["):
                    element = soup.select_one(selector)
                    if element:
                        content = self.clean_text(element)
                        results.append({
                            'selector': selector,
                            'found': True,
                            'text': content,
                            'raw': str(element)[:100]
                        })
                        print(f"{i:2d}. ✓ {selector}")
                        print(f"    Text: {content}")
                else:
                    element = soup.select_one(selector)
                    if element:
                        text = self.clean_text(element)
                        results.append({
                            'selector': selector,
                            'found': True,
                            'text': text,
                            'raw': str(element)[:100]
                        })
                        print(f"{i:2d}. ✓ {selector}")
                        print(f"    Text: {text}")
                    else:
                        print(f"{i:2d}. ✗ {selector} (not found)")
            except Exception as e:
                print(f"{i:2d}. ✗ {selector} (error: {str(e)[:50]})")
        
        return results
    
    def search_date_patterns(self, soup):
        """Search for date patterns in the entire page"""
        print("\nSearching for date patterns in page text:")
        print("-" * 50)
        
        # Get all text from the page
        all_text = soup.get_text()
        
        # Look for date patterns
        date_patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{4}',  # DD.MM.YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',     # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',     # MM/DD/YYYY
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',  # Month DD, YYYY
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'    # DD Month YYYY
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                for match in matches[:5]:  # Limit to first 5 matches
                    found_dates.append(match)
                    print(f"Found: {match}")
        
        return found_dates
    
    def inspect_html_structure(self, soup):
        """Inspect HTML structure around common date locations"""
        print("\nInspecting HTML structure:")
        print("-" * 50)
        
        # Look for elements that might contain dates
        elements_to_check = [
            ('h1', 'h1 elements'),
            ('h2', 'h2 elements'),
            ('h3', 'h3 elements'),
            ('h4', 'h4 elements'),
            ('h5', 'h5 elements'),
            ('h6', 'h6 elements'),
            ('.article-info', 'article-info class'),
            ('.article-header', 'article-header class'),
            ('.press-release-info', 'press-release-info class'),
            ('.publication-info', 'publication-info class'),
            ('header', 'header elements'),
            ('footer', 'footer elements'),
            ('.meta', 'meta class'),
            ('.post-meta', 'post-meta class')
        ]
        
        for tag, description in elements_to_check:
            elements = soup.select(tag) if '.' in tag else soup.find_all(tag)
            if elements:
                print(f"\n{description} found ({len(elements)}):")
                for i, elem in enumerate(elements[:3]):  # Show first 3
                    text = self.clean_text(elem)
                    if text and len(text) > 0:
                        print(f"  {i+1}. {text[:100]}...")
    
    def analyze_response(self):
        """Main analysis function"""
        print(f"Testing date extraction for URL:")
        print(f"{self.url}")
        print("=" * 80)
        
        try:
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 1. Test all selectors
            selector_results = self.test_selectors(soup)
            
            # 2. Search for date patterns in text
            date_patterns = self.search_date_patterns(soup)
            
            # 3. Inspect HTML structure
            self.inspect_html_structure(soup)
            
            # 4. Try to extract and parse dates
            print("\n" + "=" * 80)
            print("DATE EXTRACTION SUMMARY:")
            print("=" * 80)
            
            all_found_texts = []
            
            # Collect text from successful selectors
            for result in selector_results:
                if result['found'] and result['text']:
                    all_found_texts.append({
                        'source': 'selector',
                        'selector': result['selector'],
                        'text': result['text']
                    })
            
            # Add date patterns found
            for pattern in date_patterns:
                all_found_texts.append({
                    'source': 'pattern_match',
                    'selector': 'regex_pattern',
                    'text': pattern
                })
            
            # Try to parse dates
            parsed_dates = []
            for item in all_found_texts:
                date_obj, year = self.parse_date(item['text'])
                if date_obj:
                    parsed_dates.append({
                        'source': item['source'],
                        'selector': item['selector'],
                        'text': item['text'],
                        'date_iso': date_obj.isoformat()[:10],
                        'year': year
                    })
            
            if parsed_dates:
                print(f"\n✓ SUCCESS: Found {len(parsed_dates)} parsable dates:")
                for i, date_info in enumerate(parsed_dates, 1):
                    print(f"\n  Date {i}:")
                    print(f"    Source: {date_info['source']}")
                    print(f"    Selector: {date_info['selector']}")
                    print(f"    Raw text: {date_info['text']}")
                    print(f"    ISO date: {date_info['date_iso']}")
                    print(f"    Year: {date_info['year']}")
            else:
                print(f"\n✗ NO PARSABLE DATES FOUND")
                
                # Show what we did find
                if all_found_texts:
                    print(f"\nFound {len(all_found_texts)} potential date sources (couldn't parse):")
                    for i, item in enumerate(all_found_texts[:5], 1):
                        print(f"  {i}. [{item['source']}] {item['selector']}: {item['text'][:100]}")
                else:
                    print("No potential date sources found at all.")
            
            # 5. Save results to file
            results = {
                'url': self.url,
                'status': 'success',
                'parsed_dates': parsed_dates,
                'all_found_texts': all_found_texts[:20],  # Limit to first 20
                'timestamp': datetime.now().isoformat()
            }
            
            with open('date_test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to 'date_test_results.json'")
            
            return parsed_dates
            
        except requests.RequestException as e:
            print(f"Error fetching URL: {e}")
            return None
        except Exception as e:
            print(f"Error analyzing page: {e}")
            return None


def test_multiple_urls():
    """Test multiple example URLs"""
    test_urls = [
        # Add URLs from your output
        "https://www.press.bmwgroup.com/global/article/detail/T0454343EN",  # from-three-continents
        "https://www.press.bmwgroup.com/global/article/detail/T0454334EN",  # lian-pga-championship
        "https://www.press.bmwgroup.com/global/article/detail/T0454242EN",  # ings-compared-to-2019
        "https://www.press.bmwgroup.com/global/article/detail/T0454204EN",  # th-new-bmw-m5-touring
        # Add a few more from different sitemaps
        "https://www.press.bmwgroup.com/global/article/detail/T0454168EN",  # y-%e2%80%93-and-beyond
        "https://www.press.bmwgroup.com/global/article/detail/T0453927EN",  # fication-and-co2-aware
    ]
    
    print("Testing multiple URLs...")
    print("=" * 80)
    
    all_results = {}
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST {i}/{len(test_urls)}: {url}")
        print(f"{'='*80}")
        
        tester = BMWDateTester(url)
        results = tester.analyze_response()
        
        all_results[url] = results
        
        # Pause between requests
        if i < len(test_urls):
            print("\nWaiting 2 seconds before next request...")
            import time
            time.sleep(2)
    
    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY:")
    print("=" * 80)
    
    success_count = sum(1 for results in all_results.values() if results)
    print(f"Successful date extraction: {success_count}/{len(test_urls)}")
    
    # Save overall results
    with open('all_date_tests_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("✓ All results saved to 'all_date_tests_summary.json'")


if __name__ == "__main__":
    # Test a single URL
    single_url = "https://www.press.bmwgroup.com/global/article/detail/T0454343EN"
    
    print("BMW Press Release Date Extraction Tester")
    print("=" * 80)
    
   
    choice = input(f"Test default URL or enter custom? (default/custom): ").strip().lower()
    
    if choice == 'custom':
        custom_url = input("Enter BMW press release URL: ").strip()
        if custom_url:
            single_url = custom_url
        else:
            print("Using default URL...")
    
    # Test single URL
    tester = BMWDateTester(single_url)
    tester.analyze_response()
    
   
    print("\n" + "=" * 80)
    multi_test = input("\nDo you also want to test multiple example URLs? (yes/no): ").strip().lower()
    
    if multi_test in ['yes', 'y']:
        test_multiple_urls()
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("Check 'date_test_results.json' for detailed results.")