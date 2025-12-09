

# import requests
# import xml.etree.ElementTree as ET
# from bs4 import BeautifulSoup
# import time
# import re
# import random
# import json
# import os
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from urllib.parse import urlparse

# class BMWE2EPipeline:
#     def __init__(self, base_url="https://www.press.bmwgroup.com"):
#         self.base_url = base_url
#         self.robots_url = f"{base_url}/robots.txt"
#         self.english_sitemaps = []
#         self.all_press_urls = []
#         self.all_press_data = []
        
#         # Your specific selectors - ALL SAVED IN METADATA
#         self.selectors = {
#             'HEADLINE': "#content > div > div.content-left > div > h1",
#             'SUBHEADLINE': "#content > div > div.content-left > div > h2",
#             'AUTHOR': "#content > div > div.content-left > div > div.left > p:nth-child(4)",
#             'DATE': ".date",  # Simplified based on testing
#             'ARTICLE_BODY': "#article-text > p"
#         }
    
#     def fetch_english_sitemaps(self):
#         """Step 1: Fetch robots.txt and extract English text sitemaps"""
#         print("="*60)
#         print("STEP 1: Fetching English text sitemaps from robots.txt")
#         print("="*60)
        
#         try:
#             response = requests.get(self.robots_url, timeout=10)
#             response.raise_for_status()
#             robots_content = response.text
            
#             # Extract all sitemap URLs
#             sitemap_pattern = r'Sitemap:\s*(https?://[^\s]+)'
#             all_sitemaps = re.findall(sitemap_pattern, robots_content, re.IGNORECASE)
            
#             # Filter for English text sitemaps
#             self.english_sitemaps = [
#                 sitemap for sitemap in all_sitemaps 
#                 if sitemap.endswith('_en.xml') and 'sitemap_text' in sitemap
#             ]
            
#             print(f"Found {len(all_sitemaps)} total sitemaps in robots.txt")
#             print(f"Filtered to {len(self.english_sitemaps)} English text sitemaps:\n")
            
#             for i, sitemap in enumerate(self.english_sitemaps, 1):
#                 print(f"{i:2d}. {sitemap}")
            
#             # Save the list
#             with open('english_text_sitemaps.json', 'w') as f:
#                 json.dump(self.english_sitemaps, f, indent=2)
#             print(f"\n✓ Saved sitemap list to 'english_text_sitemaps.json'")
            
#             return self.english_sitemaps
            
#         except Exception as e:
#             print(f"Error fetching robots.txt: {e}")
#             return []
    
#     def parse_sitemap(self, sitemap_url):
#         """Parse a single sitemap to extract press release URLs"""
#         try:
#             response = requests.get(sitemap_url, timeout=10)
#             response.raise_for_status()
            
#             # Register namespace
#             namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
#             root = ET.fromstring(response.content)
#             urls = []
            
#             for url in root.findall('.//ns:loc', namespaces):
#                 press_url = url.text
#                 # Only include URLs that look like press releases
#                 if 'pressDetail' in press_url and 'EN' in press_url.upper():
#                     urls.append(press_url)
            
#             print(f"  Parsed {sitemap_url.split('/')[-1]}: Found {len(urls)} press releases")
#             return urls
            
#         except Exception as e:
#             print(f"  Error parsing {sitemap_url}: {e}")
#             return []
    
#     def clean_text(self, element):
#         """Clean text, removing ALL formatting tags"""
#         if not element:
#             return None
        
#         try:
#             element_copy = BeautifulSoup(str(element), 'html.parser')
            
#             # Remove formatting tags
#             for tag in element_copy.find_all(['strong', 'b', 'em', 'i', 'u', 'span']):
#                 if tag and hasattr(tag, 'unwrap'):
#                     tag.unwrap()
            
#             # Replace <br> tags
#             for br in element_copy.find_all("br"):
#                 if br:
#                     br.replace_with(" ")
            
#             # Get clean text
#             text = element_copy.get_text(separator=" ", strip=True)
#             if text:
#                 text = text.replace('\xa0', ' ').replace('\u200b', '')
#                 text = ' '.join(text.split())
            
#             return text
#         except Exception as e:
#             print(f"    Error cleaning text: {e}")
#             return None
    
#     def parse_date_from_text(self, date_text):
#         """Parse date from various formats including BMW's format"""
#         if not date_text:
#             return None, None
        
#         try:
#             # Test multiple date formats
#             date_formats = [
#                 # BMW format: "Fri Nov 21 10:00:00 CET 2025"
#                 (r'(\w{3})\s+(\w{3})\s+(\d{1,2})\s+\d{2}:\d{2}:\d{2}\s+\w+\s+(\d{4})', 
#                  lambda m: datetime(int(m.group(4)), self._month_to_num(m.group(2)), int(m.group(3)))),
                
#                 # European format: "21.11.2025"
#                 (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', 
#                  lambda m: datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))),
                
#                 # ISO format: "2025-11-21"
#                 (r'(\d{4})-(\d{1,2})-(\d{1,2})', 
#                  lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
                
#                 # US format: "11/21/2025"
#                 (r'(\d{1,2})/(\d{1,2})/(\d{4})', 
#                  lambda m: datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))),
                
#                 # With full month: "November 21, 2025"
#                 (r'(\w+)\s+(\d{1,2}),\s+(\d{4})', 
#                  lambda m: datetime(int(m.group(3)), self._month_to_num(m.group(1)), int(m.group(2)))),
                
#                 # With full month: "21 November 2025"
#                 (r'(\d{1,2})\s+(\w+)\s+(\d{4})', 
#                  lambda m: datetime(int(m.group(3)), self._month_to_num(m.group(2)), int(m.group(1)))),
#             ]
            
#             for pattern, parser_func in date_formats:
#                 match = re.search(pattern, date_text, re.IGNORECASE)
#                 if match:
#                     date_obj = parser_func(match)
#                     return date_obj, date_obj.year
            
#             # Try datetime.strptime for common formats
#             strptime_formats = [
#                 '%a %b %d %H:%M:%S %Z %Y',  # Fri Nov 21 10:00:00 CET 2025
#                 '%Y-%m-%d',
#                 '%d.%m.%Y',
#                 '%m/%d/%Y',
#                 '%B %d, %Y',
#                 '%d %B %Y',
#             ]
            
#             for fmt in strptime_formats:
#                 try:
#                     date_obj = datetime.strptime(date_text.strip(), fmt)
#                     return date_obj, date_obj.year
#                 except ValueError:
#                     continue
                    
#         except Exception as e:
#             print(f"    Date parsing error for '{date_text}': {e}")
        
#         return None, None
    
#     def _month_to_num(self, month_str):
#         """Convert month name or abbreviation to number"""
#         month_dict = {
#             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
#             'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
#             'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
#             'january': 1, 'february': 2, 'march': 3, 'april': 4,
#             'may': 5, 'june': 6, 'july': 7, 'august': 8,
#             'september': 9, 'october': 10, 'november': 11, 'december': 12
#         }
#         return month_dict.get(month_str.lower(), 1)
    
#     def scrape_press_release(self, url):
#         """Scrape a single press release - INCLUDING ALL ELEMENTS"""
#         try:
#             response = requests.get(url, timeout=15)
#             response.raise_for_status()
            
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Extract using selectors
#             headline_elem = soup.select_one(self.selectors['HEADLINE'])
#             subheadline_elem = soup.select_one(self.selectors['SUBHEADLINE'])
#             author_elem = soup.select_one(self.selectors['AUTHOR'])
#             date_elem = soup.select_one(self.selectors['DATE'])
#             article_body_elems = soup.select(self.selectors['ARTICLE_BODY'])
            
#             # Clean extracted text
#             headline = self.clean_text(headline_elem) if headline_elem else None
#             subheadline = self.clean_text(subheadline_elem) if subheadline_elem else None
#             author = self.clean_text(author_elem) if author_elem else None
#             date_text = self.clean_text(date_elem) if date_elem else None
            
#             # Parse date from text (handles multiple formats)
#             date_obj, year = self.parse_date_from_text(date_text) if date_text else (None, None)
#             date_iso = date_obj.isoformat()[:10] if date_obj else None  # YYYY-MM-DD format
            
#             # Combine article body paragraphs
#             article_body = ""
#             if article_body_elems:
#                 for p in article_body_elems:
#                     cleaned_p = self.clean_text(p)
#                     if cleaned_p and len(cleaned_p) > 10:  # Filter very short paragraphs
#                         article_body += cleaned_p + "\n\n"
            
#             # Extract metadata from URL
#             url_params = {}
#             if '?' in url:
#                 params = url.split('?')[1]
#                 for param in params.split('&'):
#                     if '=' in param:
#                         key, value = param.split('=', 1)
#                         url_params[key] = value
            
#             # Extract ID from URL
#             press_id = url_params.get('id', '')
            
#             return {
#                 'url': url,
#                 'headline': headline or 'No headline found',
#                 'subheadline': subheadline or '',
#                 'author': author or 'Unknown',
#                 'date_text': date_text or 'No date found',
#                 'date': date_iso,
#                 'year': year,
#                 'content': article_body.strip() if article_body else 'No content found',
#                 'press_id': press_id,
#                 'title_param': url_params.get('title', ''),
#                 'scrape_timestamp': datetime.now().isoformat(),
#                 'char_count': len(article_body) if article_body else 0,
#                 'para_count': len(article_body_elems),
#                 # Store element information for debugging
#                 'elements_found': {
#                     'headline_found': headline_elem is not None,
#                     'subheadline_found': subheadline_elem is not None,
#                     'author_found': author_elem is not None,
#                     'date_found': date_elem is not None,
#                     'body_paragraphs_found': len(article_body_elems)
#                 }
#             }
            
#         except Exception as e:
#             print(f"    Error scraping {url[-50:]}: {str(e)[:100]}")
#             return None
    
#     def collect_all_press_urls(self, max_sitemaps=None, max_per_sitemap=50):
#         """Step 2: Parse all English sitemaps to collect press release URLs"""
#         print("\n" + "="*60)
#         print("STEP 2: Collecting press release URLs from all sitemaps")
#         print("="*60)
        
#         self.all_press_urls = []
#         sitemaps_to_process = self.english_sitemaps[:max_sitemaps] if max_sitemaps else self.english_sitemaps
        
#         for i, sitemap_url in enumerate(sitemaps_to_process, 1):
#             print(f"\nProcessing sitemap {i}/{len(sitemaps_to_process)}: {sitemap_url.split('/')[-1]}")
            
#             urls = self.parse_sitemap(sitemap_url)
#             if urls:
#                 # Limit per sitemap if specified
#                 limited_urls = urls[:max_per_sitemap] if max_per_sitemap else urls
#                 self.all_press_urls.extend(limited_urls)
#                 print(f"  Added {len(limited_urls)} URLs (limited to first {max_per_sitemap})")
            
#             time.sleep(1)  # Brief pause between sitemaps
        
#         # Remove duplicates
#         self.all_press_urls = list(dict.fromkeys(self.all_press_urls))
#         print(f"\n✓ Total unique press release URLs collected: {len(self.all_press_urls)}")
        
#         # Save URL list
#         with open('all_press_urls.json', 'w') as f:
#             json.dump(self.all_press_urls, f, indent=2)
#         print(f"✓ Saved URL list to 'all_press_urls.json'")
        
#         return self.all_press_urls
    
#     def scrape_all_press_releases(self, max_releases=None, delay=2, target_year=None):
#         """Step 3: Scrape all collected press releases with optional year filtering"""
#         print("\n" + "="*60)
#         print("STEP 3: Scraping press release content")
#         print("="*60)
        
#         self.all_press_data = []
#         urls_to_scrape = self.all_press_urls[:max_releases] if max_releases else self.all_press_urls
        
#         successful = 0
#         failed = 0
#         year_matches = 0
#         no_year_info = 0
#         wrong_year = 0
        
#         print(f"Scraping {len(urls_to_scrape)} press releases" + 
#               (f" (filtering for year {target_year})" if target_year else "") + "...\n")
        
#         for i, url in enumerate(urls_to_scrape, 1):
#             print(f"  [{i}/{len(urls_to_scrape)}] {url[-80:]}...")
            
#             data = self.scrape_press_release(url)
            
#             if data:
#                 # Apply year filter if specified
#                 if target_year:
#                     if data.get('year') == target_year:
#                         self.all_press_data.append(data)
#                         year_matches += 1
#                         print(f"    ✓ {target_year}: {data['headline'][:60]}...")
#                     elif data.get('year') is None:
#                         no_year_info += 1
#                         print(f"    ? No year info")
#                     else:
#                         wrong_year += 1
#                         print(f"    ✗ Wrong year ({data.get('year')})")
#                 else:
#                     # No filtering, add all valid articles
#                     if data['content'] != 'No content found' and len(data['content']) > 100:
#                         self.all_press_data.append(data)
#                         successful += 1
#                         print(f"    ✓ Extracted: {data['headline'][:60]}...")
#                     else:
#                         failed += 1
#                         print(f"    ✗ Failed or insufficient content")
#             else:
#                 failed += 1
#                 print(f"    ✗ Scraping failed")
            
#             # Respectful delay with randomness
#             time.sleep(delay + random.uniform(0, 1))
            
#             # Save progress periodically
#             if i % 10 == 0:
#                 if target_year:
#                     status = f"[Progress: {i}/{len(urls_to_scrape)}, {target_year} matches: {year_matches}, Wrong year: {wrong_year}, No year: {no_year_info}]"
#                 else:
#                     status = f"[Progress: {i}/{len(urls_to_scrape)}, Successful: {successful}, Failed: {failed}]"
#                 print(f"    {status}")
        
#         # Final statistics
#         if target_year:
#             print(f"\n✓ Scraping complete for year {target_year}:")
#             print(f"  - Matches: {year_matches}")
#             print(f"  - Wrong year: {wrong_year}")
#             print(f"  - No year info: {no_year_info}")
#             print(f"  - Failed: {failed}")
#         else:
#             print(f"\n✓ Scraping complete: {successful} successful, {failed} failed")
        
#         # Save raw scraped data
#         if self.all_press_data:
#             with open('raw_press_data.json', 'w', encoding='utf-8') as f:
#                 json.dump(self.all_press_data, f, indent=2, ensure_ascii=False)
#             print(f"✓ Saved raw data to 'raw_press_data.json'")
        
#         return self.all_press_data
    
#     def create_datasets(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#         """Step 4: Create train/validation/test splits"""
#         print("\n" + "="*60)
#         print("STEP 4: Creating train/validation/test splits")
#         print("="*60)
        
#         if not self.all_press_data:
#             print("No data to split!")
#             return None
        
#         # Filter out articles with insufficient content
#         filtered_data = [
#             item for item in self.all_press_data 
#             if item['content'] != 'No content found' and len(item['content']) > 200
#         ]
        
#         print(f"Filtered to {len(filtered_data)} articles with sufficient content")
        
#         # Create splits
#         train_data, temp_data = train_test_split(
#             filtered_data, 
#             test_size=(1 - train_ratio),
#             random_state=42,
#             shuffle=True
#         )
        
#         # Calculate validation ratio from remaining data
#         val_size = val_ratio / (val_ratio + test_ratio)
#         val_data, test_data = train_test_split(
#             temp_data,
#             test_size=(1 - val_size),
#             random_state=42,
#             shuffle=True
#         )
        
#         datasets = {
#             'train': train_data,
#             'val': val_data,
#             'test': test_data
#         }
        
#         print(f"Training set: {len(train_data)} articles")
#         print(f"Validation set: {len(val_data)} articles")
#         print(f"Test set: {len(test_data)} articles")
        
#         return datasets
    
#     def save_datasets(self, datasets, output_dir='./bmw_press_datasets'):
#         """Step 5: Save datasets in multiple formats - INCLUDING HEADINGS"""
#         print("\n" + "="*60)
#         print("STEP 5: Saving datasets")
#         print("="*60)
        
#         os.makedirs(output_dir, exist_ok=True)
        
#         for split_name, data in datasets.items():
#             # Save as JSON (with metadata)
#             json_file = os.path.join(output_dir, f'{split_name}.json')
#             with open(json_file, 'w', encoding='utf-8') as f:
#                 json.dump(data, f, indent=2, ensure_ascii=False)
            
#             # Save as plain text - COMPLETE ARTICLE WITH HEADINGS AND METADATA
#             txt_file = os.path.join(output_dir, f'{split_name}.txt')
#             with open(txt_file, 'w', encoding='utf-8') as f:
#                 for i, item in enumerate(data, 1):
#                     # Build the complete article text
#                     article_text = []
                    
#                     # Add headline (MAIN TITLE)
#                     if item['headline'] and item['headline'] != 'No headline found':
#                         article_text.append(f"# {item['headline']}")
                    
#                     # Add subheadline if exists
#                     if item['subheadline'] and item['subheadline'].strip():
#                         article_text.append(f"## {item['subheadline']}")
                    
#                     # Add metadata line
#                     metadata_parts = []
#                     if item.get('date'):
#                         metadata_parts.append(f"Date: {item['date']}")
#                     if item['author'] and item['author'] != 'Unknown':
#                         metadata_parts.append(f"Author: {item['author']}")
#                     if item.get('press_id'):
#                         metadata_parts.append(f"ID: {item['press_id']}")
                    
#                     if metadata_parts:
#                         article_text.append(" | ".join(metadata_parts))
                    
#                     # Add separator
#                     article_text.append("-" * 80)
                    
#                     # Add content
#                     if item['content'] and item['content'] != 'No content found':
#                         article_text.append(item['content'])
                    
#                     # Write to file with separator between articles
#                     f.write("\n".join(article_text))
#                     f.write("\n\n" + "="*80 + "\n\n")
            
#             # Save as plain text - HEADINGS ONLY (for analysis)
#             txt_headings_file = os.path.join(output_dir, f'{split_name}_headings.txt')
#             with open(txt_headings_file, 'w', encoding='utf-8') as f:
#                 for i, item in enumerate(data, 1):
#                     heading_entry = []
#                     if item['headline'] and item['headline'] != 'No headline found':
#                         heading_entry.append(f"H1: {item['headline']}")
#                     if item['subheadline'] and item['subheadline'].strip():
#                         heading_entry.append(f"H2: {item['subheadline']}")
                    
#                     if heading_entry:
#                         f.write("\n".join(heading_entry) + "\n---\n")
            
#             # Save as JSONL (one JSON object per line)
#             jsonl_file = os.path.join(output_dir, f'{split_name}.jsonl')
#             with open(jsonl_file, 'w', encoding='utf-8') as f:
#                 for item in data:
#                     f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
#             # Save as CSV for analysis
#             csv_file = os.path.join(output_dir, f'{split_name}.csv')
#             with open(csv_file, 'w', encoding='utf-8', newline='') as f:
#                 import csv
#                 writer = csv.writer(f)
#                 # Write header
#                 writer.writerow(['headline', 'subheadline', 'author', 'date', 'url', 'press_id', 'char_count', 'para_count'])
#                 # Write data
#                 for item in data:
#                     writer.writerow([
#                         item['headline'],
#                         item['subheadline'],
#                         item['author'],
#                         item.get('date', ''),
#                         item['url'],
#                         item['press_id'],
#                         item['char_count'],
#                         item['para_count']
#                     ])
            
#             print(f"✓ {split_name.upper()}:")
#             print(f"  - JSON (full metadata): {json_file}")
#             print(f"  - Text (complete articles): {txt_file}")
#             print(f"  - Text (headings only): {txt_headings_file}")
#             print(f"  - JSONL: {jsonl_file}")
#             print(f"  - CSV: {csv_file}")
        
#         # Save metadata with ALL selectors
#         metadata = {
#             'total_sitemaps': len(self.english_sitemaps),
#             'total_urls_collected': len(self.all_press_urls),
#             'total_articles_scraped': len(self.all_press_data),
#             'train_size': len(datasets['train']),
#             'val_size': len(datasets['val']),
#             'test_size': len(datasets['test']),
#             'selectors_used': self.selectors,
#             'selector_descriptions': {
#                 'HEADLINE': 'Main headline/title of the press release',
#                 'SUBHEADLINE': 'Subheading/secondary title',
#                 'AUTHOR': 'Author/creator of the press release',
#                 'DATE': 'Publication date',
#                 'ARTICLE_BODY': 'Main content paragraphs'
#             },
#             'date_selector_info': 'Using ".date" selector based on successful testing',
#             'pipeline_execution_date': datetime.now().isoformat(),
#             'source_robots_txt': self.robots_url,
#             'base_url': self.base_url,
#             'output_formats': ['JSON', 'TXT (complete)', 'TXT (headings only)', 'JSONL', 'CSV']
#         }
        
#         metadata_file = os.path.join(output_dir, 'metadata.json')
#         with open(metadata_file, 'w') as f:
#             json.dump(metadata, f, indent=2)
        
#         print(f"\n✓ Metadata saved to {metadata_file}")
#         print(f"✓ All selectors saved in metadata:")
#         for selector, path in self.selectors.items():
#             print(f"  - {selector}: {path}")
        
#         return output_dir
    
#     def run_complete_pipeline(self, 
#                             max_sitemaps=None, 
#                             max_per_sitemap=50,
#                             max_releases=None,
#                             delay=2,
#                             target_year=None):
#         """Run the complete end-to-end pipeline with optional year filtering"""
#         print("\n" + "="*60)
#         title = "BMW PRESS RELEASE COMPLETE PIPELINE"
#         if target_year:
#             title += f" (Filtering for {target_year})"
#         print(title)
#         print("="*60)
        
#         # Display all selectors being used
#         print("\nSELECTORS BEING USED:")
#         print("-" * 40)
#         for selector, path in self.selectors.items():
#             print(f"{selector}: {path}")
#         print("-" * 40 + "\n")
        
#         start_time = datetime.now()
        
#         # Step 1: Get English sitemaps
#         self.fetch_english_sitemaps()
        
#         if not self.english_sitemaps:
#             print("No English sitemaps found. Exiting.")
#             return
        
#         # Step 2: Collect all press release URLs
#         self.collect_all_press_urls(
#             max_sitemaps=max_sitemaps,
#             max_per_sitemap=max_per_sitemap
#         )
        
#         if not self.all_press_urls:
#             print("No press release URLs collected. Exiting.")
#             return
        
#         # Step 3: Scrape all press releases WITH YEAR FILTERING
#         self.scrape_all_press_releases(
#             max_releases=max_releases,
#             delay=delay,
#             target_year=target_year
#         )
        
#         if not self.all_press_data:
#             year_msg = f" for year {target_year}" if target_year else ""
#             print(f"No press release data scraped{year_msg}. Exiting.")
#             return
        
#         # Step 4: Create datasets
#         datasets = self.create_datasets()
        
#         if not datasets:
#             print("Failed to create datasets. Exiting.")
#             return
        
#         # Step 5: Save datasets
#         output_dir = self.save_datasets(datasets)
        
#         # Final summary
#         end_time = datetime.now()
#         duration = (end_time - start_time).total_seconds() / 60
        
#         print("\n" + "="*60)
#         print("PIPELINE COMPLETE - SUMMARY")
#         print("="*60)
#         print(f"Total English sitemaps: {len(self.english_sitemaps)}")
#         print(f"Total press release URLs: {len(self.all_press_urls)}")
#         print(f"Successfully scraped articles: {len(self.all_press_data)}")
#         if target_year:
#             print(f"Articles from {target_year}: {len(self.all_press_data)}")
#         print(f"Training set: {len(datasets['train'])} articles")
#         print(f"Validation set: {len(datasets['val'])} articles")
#         print(f"Test set: {len(datasets['test'])} articles")
#         print(f"Total pipeline time: {duration:.2f} minutes")
#         print(f"Output directory: {output_dir}")
        
#         # Show sample of articles with ALL metadata
#         if self.all_press_data:
#             print(f"\nSAMPLE ARTICLES (with all metadata):")
#             print("-"*60)
#             for i, article in enumerate(self.all_press_data[:3]):
#                 print(f"\nArticle {i+1}:")
#                 print(f"  Headline: {article['headline'][:80]}...")
#                 if article['subheadline']:
#                     print(f"  Subheadline: {article['subheadline'][:60]}...")
#                 print(f"  Author: {article['author']}")
#                 print(f"  Date: {article.get('date', 'N/A')}")
#                 print(f"  Content length: {article['char_count']} chars, {article['para_count']} paragraphs")
        
#         return datasets


# # Main execution with updated date parsing
# if __name__ == "__main__":
#     # Initialize pipeline
#     pipeline = BMWE2EPipeline()
    
#     # Run pipeline for 2025 articles only
#     datasets = pipeline.run_complete_pipeline(
#         max_sitemaps=10,           # Process first 3 sitemaps (set to None for all)
#         max_per_sitemap=200,       # Take first 20 URLs from each sitemap
#         max_releases=1000,         # Scrape up to 250 press releases total
#         delay=1.5,                # Delay between requests (be respectful!)
#         target_year=2025          # Filter for 2025 articles only
#     )
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import time
import re
import random
import json
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from tqdm import tqdm  # Added for progress bar

class BMWE2EPipeline:
    def __init__(self, base_url="https://www.press.bmwgroup.com"):
        self.base_url = base_url
        self.robots_url = f"{base_url}/robots.txt"
        self.english_sitemaps = []
        self.all_press_urls = []
        self.all_press_data = []
        
        # Your specific selectors - ALL SAVED IN METADATA
        self.selectors = {
            'HEADLINE': "#content > div > div.content-left > div > h1",
            'SUBHEADLINE': "#content > div > div.content-left > div > h2",
            'AUTHOR': "#content > div > div.content-left > div > div.left > p:nth-child(4)",
            'DATE': ".date",  # Simplified based on testing
            'ARTICLE_BODY': "#article-text > p"
        }
    
    
    def fetch_english_sitemaps(self):
        """Step 1: Fetch robots.txt and extract English text sitemaps"""
        print("="*60)
        print("STEP 1: Fetching English text sitemaps from robots.txt")
        print("="*60)
        
        try:
            response = requests.get(self.robots_url, timeout=10)
            response.raise_for_status()
            robots_content = response.text
            
            # Extract all sitemap URLs
            sitemap_pattern = r'Sitemap:\s*(https?://[^\s]+)'
            all_sitemaps = re.findall(sitemap_pattern, robots_content, re.IGNORECASE)
            
            # SIMPLE AND EFFECTIVE: Filter for English text sitemaps
            self.english_sitemaps = [
                sitemap for sitemap in all_sitemaps 
                if 'sitemap_text' in sitemap.lower()  # Must be a text sitemap
                and '_en' in sitemap.lower()          # Must contain '_en' somewhere
            ]
            
            print(f"Found {len(all_sitemaps)} total sitemaps in robots.txt")
            print(f"Filtered to {len(self.english_sitemaps)} English text sitemaps:\n")
            
            for i, sitemap in enumerate(self.english_sitemaps, 1):
                print(f"{i:2d}. {sitemap}")
            
            # Debug: List the matched patterns
            print(f"\nMatched sitemaps with '_en' pattern:")
            for sitemap in self.english_sitemaps:
                filename = sitemap.split('/')[-1]
                # Highlight where '_en' appears
                highlighted = filename.replace('_en', '[_en]')
                print(f"  - {highlighted}")
            
            # Save the list
            with open('english_text_sitemaps.json', 'w') as f:
                json.dump(self.english_sitemaps, f, indent=2)
            print(f"\n✓ Saved sitemap list to 'english_text_sitemaps.json'")
            
            return self.english_sitemaps
            
        except Exception as e:
            print(f"Error fetching robots.txt: {e}")
            return []
        

    def parse_sitemap(self, sitemap_url):
        """Parse a single sitemap to extract press release URLs"""
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            
            # Register namespace
            namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            root = ET.fromstring(response.content)
            urls = []
            
            for url in root.findall('.//ns:loc', namespaces):
                press_url = url.text
                # Only include URLs that look like press releases
                if 'pressDetail' in press_url and 'EN' in press_url.upper():
                    urls.append(press_url)
            
            print(f"  Parsed {sitemap_url.split('/')[-1]}: Found {len(urls)} press releases")
            return urls
            
        except Exception as e:
            print(f"  Error parsing {sitemap_url}: {e}")
            return []
    
    def clean_text(self, element):
        """Clean text, removing ALL formatting tags"""
        if not element:
            return None
        
        try:
            element_copy = BeautifulSoup(str(element), 'html.parser')
            
            # Remove formatting tags
            for tag in element_copy.find_all(['strong', 'b', 'em', 'i', 'u', 'span']):
                if tag and hasattr(tag, 'unwrap'):
                    tag.unwrap()
            
            # Replace <br> tags
            for br in element_copy.find_all("br"):
                if br:
                    br.replace_with(" ")
            
            # Get clean text
            text = element_copy.get_text(separator=" ", strip=True)
            if text:
                text = text.replace('\xa0', ' ').replace('\u200b', '')
                text = ' '.join(text.split())
            
            return text
        except Exception as e:
            print(f"    Error cleaning text: {e}")
            return None
    
    def parse_date_from_text(self, date_text):
        """Parse date from various formats including BMW's format"""
        if not date_text:
            return None, None
        
        try:
            # Test multiple date formats
            date_formats = [
                # BMW format: "Fri Nov 21 10:00:00 CET 2025"
                (r'(\w{3})\s+(\w{3})\s+(\d{1,2})\s+\d{2}:\d{2}:\d{2}\s+\w+\s+(\d{4})', 
                 lambda m: datetime(int(m.group(4)), self._month_to_num(m.group(2)), int(m.group(3)))),
                
                # European format: "21.11.2025"
                (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', 
                 lambda m: datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))),
                
                # ISO format: "2025-11-21"
                (r'(\d{4})-(\d{1,2})-(\d{1,2})', 
                 lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
                
                # US format: "11/21/2025"
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', 
                 lambda m: datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))),
                
                # With full month: "November 21, 2025"
                (r'(\w+)\s+(\d{1,2}),\s+(\d{4})', 
                 lambda m: datetime(int(m.group(3)), self._month_to_num(m.group(1)), int(m.group(2)))),
                
                # With full month: "21 November 2025"
                (r'(\d{1,2})\s+(\w+)\s+(\d{4})', 
                 lambda m: datetime(int(m.group(3)), self._month_to_num(m.group(2)), int(m.group(1)))),
            ]
            
            for pattern, parser_func in date_formats:
                match = re.search(pattern, date_text, re.IGNORECASE)
                if match:
                    date_obj = parser_func(match)
                    return date_obj, date_obj.year
            
            # Try datetime.strptime for common formats
            strptime_formats = [
                '%a %b %d %H:%M:%S %Z %Y',  # Fri Nov 21 10:00:00 CET 2025
                '%Y-%m-%d',
                '%d.%m.%Y',
                '%m/%d/%Y',
                '%B %d, %Y',
                '%d %B %Y',
            ]
            
            for fmt in strptime_formats:
                try:
                    date_obj = datetime.strptime(date_text.strip(), fmt)
                    return date_obj, date_obj.year
                except ValueError:
                    continue
                    
        except Exception as e:
            print(f"    Date parsing error for '{date_text}': {e}")
        
        return None, None
    
    def _month_to_num(self, month_str):
        """Convert month name or abbreviation to number"""
        month_dict = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        return month_dict.get(month_str.lower(), 1)
    
    def scrape_press_release(self, url):
        """Scrape a single press release - INCLUDING ALL ELEMENTS"""
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract using selectors
            headline_elem = soup.select_one(self.selectors['HEADLINE'])
            subheadline_elem = soup.select_one(self.selectors['SUBHEADLINE'])
            author_elem = soup.select_one(self.selectors['AUTHOR'])
            date_elem = soup.select_one(self.selectors['DATE'])
            article_body_elems = soup.select(self.selectors['ARTICLE_BODY'])
            
            # Clean extracted text
            headline = self.clean_text(headline_elem) if headline_elem else None
            subheadline = self.clean_text(subheadline_elem) if subheadline_elem else None
            author = self.clean_text(author_elem) if author_elem else None
            date_text = self.clean_text(date_elem) if date_elem else None
            
            # Parse date from text (handles multiple formats)
            date_obj, year = self.parse_date_from_text(date_text) if date_text else (None, None)
            date_iso = date_obj.isoformat()[:10] if date_obj else None  # YYYY-MM-DD format
            
            # Combine article body paragraphs
            article_body = ""
            if article_body_elems:
                for p in article_body_elems:
                    cleaned_p = self.clean_text(p)
                    if cleaned_p and len(cleaned_p) > 10:  # Filter very short paragraphs
                        article_body += cleaned_p + "\n\n"
            
            # Extract metadata from URL
            url_params = {}
            if '?' in url:
                params = url.split('?')[1]
                for param in params.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        url_params[key] = value
            
            # Extract ID from URL
            press_id = url_params.get('id', '')
            
            return {
                'url': url,
                'headline': headline or 'No headline found',
                'subheadline': subheadline or '',
                'author': author or 'Unknown',
                'date_text': date_text or 'No date found',
                'date': date_iso,
                'year': year,
                'content': article_body.strip() if article_body else 'No content found',
                'press_id': press_id,
                'title_param': url_params.get('title', ''),
                'scrape_timestamp': datetime.now().isoformat(),
                'char_count': len(article_body) if article_body else 0,
                'para_count': len(article_body_elems),
                # Store element information for debugging
                'elements_found': {
                    'headline_found': headline_elem is not None,
                    'subheadline_found': subheadline_elem is not None,
                    'author_found': author_elem is not None,
                    'date_found': date_elem is not None,
                    'body_paragraphs_found': len(article_body_elems)
                }
            }
            
        except Exception as e:
            # Don't print error here - let the progress bar handle it
            return None
    
    def collect_all_press_urls(self, max_sitemaps=None, max_per_sitemap=50):
        """Step 2: Parse all English sitemaps to collect press release URLs"""
        print("\n" + "="*60)
        print("STEP 2: Collecting press release URLs from all sitemaps")
        print("="*60)
        
        self.all_press_urls = []
        sitemaps_to_process = self.english_sitemaps[:max_sitemaps] if max_sitemaps else self.english_sitemaps
        
        print(f"Processing {len(sitemaps_to_process)} sitemaps...")
        
        for i, sitemap_url in enumerate(sitemaps_to_process, 1):
            print(f"\nProcessing sitemap {i}/{len(sitemaps_to_process)}: {sitemap_url.split('/')[-1]}")
            
            urls = self.parse_sitemap(sitemap_url)
            if urls:
                # Limit per sitemap if specified
                limited_urls = urls[:max_per_sitemap] if max_per_sitemap else urls
                self.all_press_urls.extend(limited_urls)
                print(f"  Added {len(limited_urls)} URLs (limited to first {max_per_sitemap})")
            
            time.sleep(1)  # Brief pause between sitemaps
        
        # Remove duplicates
        self.all_press_urls = list(dict.fromkeys(self.all_press_urls))
        print(f"\n✓ Total unique press release URLs collected: {len(self.all_press_urls)}")
        
        # Save URL list
        with open('all_press_urls.json', 'w') as f:
            json.dump(self.all_press_urls, f, indent=2)
        print(f"✓ Saved URL list to 'all_press_urls.json'")
        
        return self.all_press_urls
    
    def scrape_all_press_releases(self, max_releases=None, delay=2, target_year=None):
        """Step 3: Scrape all collected press releases with optional year filtering - WITH TQDM"""
        print("\n" + "="*60)
        print("STEP 3: Scraping press release content")
        print("="*60)
        
        self.all_press_data = []
        urls_to_scrape = self.all_press_urls[:max_releases] if max_releases else self.all_press_urls
        
        successful = 0
        failed = 0
        year_matches = 0
        no_year_info = 0
        wrong_year = 0
        
        print(f"Scraping {len(urls_to_scrape)} press releases" + 
              (f" (filtering for year {target_year})" if target_year else "") + "...\n")
        
        # Calculate estimated total time
        avg_delay = delay + 0.5  # Average of random.uniform(0, 1)
        est_total_seconds = len(urls_to_scrape) * avg_delay
        est_total_time = str(timedelta(seconds=int(est_total_seconds)))
        
        print(f"Estimated scraping time: ~{est_total_time} (at {avg_delay:.1f}s per URL)\n")
        
        # Create progress bar with custom formatting
        pbar = tqdm(
            urls_to_scrape, 
            desc="Scraping Press Releases", 
            unit="article",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        start_time = datetime.now()
        last_update_time = start_time
        
        for url in pbar:
            data = self.scrape_press_release(url)
            
            if data:
                # Apply year filter if specified
                if target_year:
                    if data.get('year') == target_year:
                        self.all_press_data.append(data)
                        year_matches += 1
                        status_symbol = "✓"
                        status_text = f"{target_year} match"
                    elif data.get('year') is None:
                        no_year_info += 1
                        status_symbol = "?"
                        status_text = "No year"
                    else:
                        wrong_year += 1
                        status_symbol = "✗"
                        status_text = f"Wrong year ({data.get('year')})"
                else:
                    # No filtering, add all valid articles
                    if data['content'] != 'No content found' and len(data['content']) > 100:
                        self.all_press_data.append(data)
                        successful += 1
                        status_symbol = "✓"
                        status_text = "Extracted"
                    else:
                        failed += 1
                        status_symbol = "✗"
                        status_text = "Insufficient content"
            else:
                failed += 1
                status_symbol = "✗"
                status_text = "Failed"
            
            # Update progress bar postfix with current stats
            if target_year:
                postfix_info = f"{status_symbol} {year_matches}/{wrong_year}/{no_year_info}/{failed}"
                pbar.set_postfix_str(f"Matches/Wrong/NoYear/Failed: {postfix_info}")
            else:
                postfix_info = f"{status_symbol} {successful}/{failed}"
                pbar.set_postfix_str(f"Success/Failed: {postfix_info}")
            
            # Update progress bar description with short headline if successful
            if data and data.get('headline'):
                short_headline = data['headline'][:40] + "..." if len(data['headline']) > 40 else data['headline']
                pbar.set_description(f"Scraping: {short_headline}")
            
            # Respectful delay with randomness
            time.sleep(delay + random.uniform(0, 1))
            
            # Update ETA every 10 items
            current_time = datetime.now()
            if (current_time - last_update_time).total_seconds() > 10:
                elapsed = (current_time - start_time).total_seconds()
                items_done = pbar.n
                if items_done > 0:
                    time_per_item = elapsed / items_done
                    remaining_items = len(urls_to_scrape) - items_done
                    eta_seconds = remaining_items * time_per_item
                    eta_time = str(timedelta(seconds=int(eta_seconds)))
                    pbar.set_postfix_str(f"ETA: {eta_time}", refresh=False)
                last_update_time = current_time
        
        pbar.close()
        
        # Final statistics
        if target_year:
            print(f"\n✓ Scraping complete for year {target_year}:")
            print(f"  - Matches: {year_matches}")
            print(f"  - Wrong year: {wrong_year}")
            print(f"  - No year info: {no_year_info}")
            print(f"  - Failed: {failed}")
        else:
            print(f"\n✓ Scraping complete: {successful} successful, {failed} failed")
        
        # Save raw scraped data
        if self.all_press_data:
            with open('raw_press_data.json', 'w', encoding='utf-8') as f:
                json.dump(self.all_press_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved raw data to 'raw_press_data.json'")
        
        # Print actual time taken
        actual_time = datetime.now() - start_time
        print(f"Actual scraping time: {str(actual_time).split('.')[0]}")
        
        return self.all_press_data
    
    def create_datasets(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Step 4: Create train/validation/test splits"""
        print("\n" + "="*60)
        print("STEP 4: Creating train/validation/test splits")
        print("="*60)
        
        if not self.all_press_data:
            print("No data to split!")
            return None
        
        # Filter out articles with insufficient content
        filtered_data = [
            item for item in self.all_press_data 
            if item['content'] != 'No content found' and len(item['content']) > 200
        ]
        
        print(f"Filtered to {len(filtered_data)} articles with sufficient content")
        
        # Create splits
        train_data, temp_data = train_test_split(
            filtered_data, 
            test_size=(1 - train_ratio),
            random_state=42,
            shuffle=True
        )
        
        # Calculate validation ratio from remaining data
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=42,
            shuffle=True
        )
        
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        print(f"Training set: {len(train_data)} articles")
        print(f"Validation set: {len(val_data)} articles")
        print(f"Test set: {len(test_data)} articles")
        
        return datasets
    
    def save_datasets(self, datasets, output_dir='./bmw_press_datasets'):
        """Step 5: Save datasets in multiple formats - INCLUDING HEADINGS"""
        print("\n" + "="*60)
        print("STEP 5: Saving datasets")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Add tqdm for saving progress
        for split_name, data in datasets.items():
            print(f"\nSaving {split_name} dataset ({len(data)} articles)...")
            
            # Save as JSON (with metadata)
            json_file = os.path.join(output_dir, f'{split_name}.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save as plain text - COMPLETE ARTICLE WITH HEADINGS AND METADATA
            txt_file = os.path.join(output_dir, f'{split_name}.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                for i, item in tqdm(enumerate(data, 1), total=len(data), desc=f"Saving {split_name}.txt", unit="article"):
                    # Build the complete article text
                    article_text = []
                    
                    # Add headline (MAIN TITLE)
                    if item['headline'] and item['headline'] != 'No headline found':
                        article_text.append(f"# {item['headline']}")
                    
                    # Add subheadline if exists
                    if item['subheadline'] and item['subheadline'].strip():
                        article_text.append(f"## {item['subheadline']}")
                    
                    # Add metadata line
                    metadata_parts = []
                    if item.get('date'):
                        metadata_parts.append(f"Date: {item['date']}")
                    if item['author'] and item['author'] != 'Unknown':
                        metadata_parts.append(f"Author: {item['author']}")
                    if item.get('press_id'):
                        metadata_parts.append(f"ID: {item['press_id']}")
                    
                    if metadata_parts:
                        article_text.append(" | ".join(metadata_parts))
                    
                    # Add separator
                    article_text.append("-" * 80)
                    
                    # Add content
                    if item['content'] and item['content'] != 'No content found':
                        article_text.append(item['content'])
                    
                    # Write to file with separator between articles
                    f.write("\n".join(article_text))
                    f.write("\n\n" + "="*80 + "\n\n")
            
            # Save as plain text - HEADINGS ONLY (for analysis)
            txt_headings_file = os.path.join(output_dir, f'{split_name}_headings.txt')
            with open(txt_headings_file, 'w', encoding='utf-8') as f:
                for i, item in enumerate(data, 1):
                    heading_entry = []
                    if item['headline'] and item['headline'] != 'No headline found':
                        heading_entry.append(f"H1: {item['headline']}")
                    if item['subheadline'] and item['subheadline'].strip():
                        heading_entry.append(f"H2: {item['subheadline']}")
                    
                    if heading_entry:
                        f.write("\n".join(heading_entry) + "\n---\n")
            
            # Save as JSONL (one JSON object per line)
            jsonl_file = os.path.join(output_dir, f'{split_name}.jsonl')
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for item in tqdm(data, desc=f"Saving {split_name}.jsonl", unit="article"):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Save as CSV for analysis
            csv_file = os.path.join(output_dir, f'{split_name}.csv')
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                import csv
                writer = csv.writer(f)
                # Write header
                writer.writerow(['headline', 'subheadline', 'author', 'date', 'url', 'press_id', 'char_count', 'para_count'])
                # Write data
                for item in tqdm(data, desc=f"Saving {split_name}.csv", unit="article"):
                    writer.writerow([
                        item['headline'],
                        item['subheadline'],
                        item['author'],
                        item.get('date', ''),
                        item['url'],
                        item['press_id'],
                        item['char_count'],
                        item['para_count']
                    ])
            
            print(f"✓ {split_name.upper()}:")
            print(f"  - JSON (full metadata): {json_file}")
            print(f"  - Text (complete articles): {txt_file}")
            print(f"  - Text (headings only): {txt_headings_file}")
            print(f"  - JSONL: {jsonl_file}")
            print(f"  - CSV: {csv_file}")
        
        # Save metadata with ALL selectors
        metadata = {
            'total_sitemaps': len(self.english_sitemaps),
            'total_urls_collected': len(self.all_press_urls),
            'total_articles_scraped': len(self.all_press_data),
            'train_size': len(datasets['train']),
            'val_size': len(datasets['val']),
            'test_size': len(datasets['test']),
            'selectors_used': self.selectors,
            'selector_descriptions': {
                'HEADLINE': 'Main headline/title of the press release',
                'SUBHEADLINE': 'Subheading/secondary title',
                'AUTHOR': 'Author/creator of the press release',
                'DATE': 'Publication date',
                'ARTICLE_BODY': 'Main content paragraphs'
            },
            'date_selector_info': 'Using ".date" selector based on successful testing',
            'pipeline_execution_date': datetime.now().isoformat(),
            'source_robots_txt': self.robots_url,
            'base_url': self.base_url,
            'output_formats': ['JSON', 'TXT (complete)', 'TXT (headings only)', 'JSONL', 'CSV']
        }
        
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Metadata saved to {metadata_file}")
        print(f"✓ All selectors saved in metadata:")
        for selector, path in self.selectors.items():
            print(f"  - {selector}: {path}")
        
        return output_dir
    
    def run_complete_pipeline(self, 
                            max_sitemaps=None, 
                            max_per_sitemap=50,
                            max_releases=None,
                            delay=2,
                            target_year=None):
        """Run the complete end-to-end pipeline with optional year filtering"""
        print("\n" + "="*60)
        title = "BMW PRESS RELEASE COMPLETE PIPELINE"
        if target_year:
            title += f" (Filtering for {target_year})"
        print(title)
        print("="*60)
        
        # Display all selectors being used
        print("\nSELECTORS BEING USED:")
        print("-" * 40)
        for selector, path in self.selectors.items():
            print(f"{selector}: {path}")
        print("-" * 40 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Get English sitemaps
        self.fetch_english_sitemaps()
        
        if not self.english_sitemaps:
            print("No English sitemaps found. Exiting.")
            return
        
        # Step 2: Collect all press release URLs
        self.collect_all_press_urls(
            max_sitemaps=max_sitemaps,
            max_per_sitemap=max_per_sitemap
        )
        
        if not self.all_press_urls:
            print("No press release URLs collected. Exiting.")
            return
        
        # Step 3: Scrape all press releases WITH YEAR FILTERING
        self.scrape_all_press_releases(
            max_releases=max_releases,
            delay=delay,
            target_year=target_year
        )
        
        if not self.all_press_data:
            year_msg = f" for year {target_year}" if target_year else ""
            print(f"No press release data scraped{year_msg}. Exiting.")
            return
        
        # Step 4: Create datasets
        datasets = self.create_datasets()
        
        if not datasets:
            print("Failed to create datasets. Exiting.")
            return
        
        # Step 5: Save datasets
        output_dir = self.save_datasets(datasets)
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE - SUMMARY")
        print("="*60)
        print(f"Total English sitemaps: {len(self.english_sitemaps)}")
        print(f"Total press release URLs: {len(self.all_press_urls)}")
        print(f"Successfully scraped articles: {len(self.all_press_data)}")
        if target_year:
            print(f"Articles from {target_year}: {len(self.all_press_data)}")
        print(f"Training set: {len(datasets['train'])} articles")
        print(f"Validation set: {len(datasets['val'])} articles")
        print(f"Test set: {len(datasets['test'])} articles")
        print(f"Total pipeline time: {duration:.2f} minutes")
        print(f"Output directory: {output_dir}")
        
        # Show sample of articles with ALL metadata
        if self.all_press_data:
            print(f"\nSAMPLE ARTICLES (with all metadata):")
            print("-"*60)
            for i, article in enumerate(self.all_press_data[:3]):
                print(f"\nArticle {i+1}:")
                print(f"  Headline: {article['headline'][:80]}...")
                if article['subheadline']:
                    print(f"  Subheadline: {article['subheadline'][:60]}...")
                print(f"  Author: {article['author']}")
                print(f"  Date: {article.get('date', 'N/A')}")
                print(f"  Content length: {article['char_count']} chars, {article['para_count']} paragraphs")
        
        return datasets


# Main execution with updated date parsing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = BMWE2EPipeline()
    
    # Run pipeline for 2025 articles only
    datasets = pipeline.run_complete_pipeline(
        max_sitemaps=None,           # Process first 10 sitemaps (set to None for all)
        max_per_sitemap=500,       # Take first 200 URLs from each sitemap
        max_releases=1000,         # Scrape up to 1000 press releases total
        delay=1.5,                 # Delay between requests (be respectful!)
        target_year=2025           # Filter for 2025 articles only
    )

