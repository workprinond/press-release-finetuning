import requests
import xml.etree.ElementTree as ET
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

class SitemapURLCounter:
    def __init__(self):
        self.sitemap_urls = [
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_pcgl_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_us_en_us.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_gb_en_gb.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_rr_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_ac_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_ade_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_ea_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_ca_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_in_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_ie_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_me_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_nz_en.xml',
            'https://www.press.bmwgroup.com/sitemaps/sitemap_text_za_en.xml'
        ]
    
    def parse_sitemap(self, sitemap_url):
        """Parse a single sitemap and count press release URLs"""
        try:
            response = requests.get(sitemap_url, timeout=15)
            response.raise_for_status()
            
            # Register namespace
            namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            root = ET.fromstring(response.content)
            
            # Count all press release URLs
            press_urls = []
            all_urls = []
            
            for url in root.findall('.//ns:loc', namespaces):
                url_text = url.text
                all_urls.append(url_text)
                
                # Check if it's a press release (contains 'pressDetail' and 'EN')
                if 'pressDetail' in url_text and 'EN' in url_text.upper():
                    press_urls.append(url_text)
            
            # Also check for nested sitemaps (some sitemaps contain other sitemaps)
            nested_sitemaps = []
            for url in all_urls:
                if url.endswith('.xml') and url != sitemap_url:
                    nested_sitemaps.append(url)
            
            return {
                'sitemap_name': sitemap_url.split('/')[-1],
                'total_urls': len(all_urls),
                'press_urls': press_urls,
                'nested_sitemaps': nested_sitemaps,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'sitemap_name': sitemap_url.split('/')[-1],
                'total_urls': 0,
                'press_urls': [],
                'nested_sitemaps': [],
                'status': f'error: {str(e)[:100]}',
                'error': str(e)
            }
    
    def threaded_count_all(self, max_workers=5):
        """Count URLs from all sitemaps using threading for speed"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sitemap = {
                executor.submit(self.parse_sitemap, url): url 
                for url in self.sitemap_urls
            }
            
            for future in as_completed(future_to_sitemap):
                sitemap_url = future_to_sitemap[future]
                try:
                    result = future.result()
                    results[sitemap_url] = result
                    print(f"✓ Parsed {result['sitemap_name']}: {len(result['press_urls'])} press URLs")
                except Exception as e:
                    print(f"✗ Failed {sitemap_url}: {e}")
                    results[sitemap_url] = {
                        'sitemap_name': sitemap_url.split('/')[-1],
                        'total_urls': 0,
                        'press_urls': [],
                        'status': 'failed'
                    }
                
                time.sleep(0.5)  # Brief pause between requests
        
        return results
    
    def calculate_totals(self, results):
        """Calculate summary statistics"""
        total_press_urls = 0
        total_all_urls = 0
        successful_sitemaps = 0
        failed_sitemaps = 0
        
        press_urls_by_sitemap = {}
        all_urls_list = []
        
        for sitemap_url, data in results.items():
            if data['status'] == 'success':
                successful_sitemaps += 1
                total_press_urls += len(data['press_urls'])
                total_all_urls += data['total_urls']
                all_urls_list.extend(data['press_urls'])
                press_urls_by_sitemap[data['sitemap_name']] = len(data['press_urls'])
            else:
                failed_sitemaps += 1
        
        # Remove duplicates
        unique_press_urls = list(set(all_urls_list))
        
        return {
            'successful_sitemaps': successful_sitemaps,
            'failed_sitemaps': failed_sitemaps,
            'total_press_urls_raw': total_press_urls,
            'total_press_urls_unique': len(unique_press_urls),
            'total_all_urls': total_all_urls,
            'press_urls_by_sitemap': press_urls_by_sitemap,
            'unique_press_urls': unique_press_urls,
            'avg_press_per_sitemap': total_press_urls / max(1, successful_sitemaps)
        }
    
    def estimate_scraping_time(self, total_urls, delay_per_url=1.5):
        """Estimate total scraping time based on URL count"""
        # Realistic time per URL including overhead
        processing_time_per_url = 0.5  # seconds for parsing/processing
        total_seconds_per_url = delay_per_url + processing_time_per_url
        
        total_seconds = total_urls * total_seconds_per_url
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60
        
        # Breakdown for different batch sizes
        time_estimates = {
            'per_100_urls': (100 * total_seconds_per_url) / 60,  # minutes
            'per_500_urls': (500 * total_seconds_per_url) / 3600,  # hours
            'total': {
                'seconds': total_seconds,
                'minutes': total_minutes,
                'hours': total_hours,
                'days': total_hours / 24
            }
        }
        
        return time_estimates
    
    def run_analysis(self):
        """Run complete analysis"""
        print("="*70)
        print("BMW PRESS SITEMAP URL ANALYSIS")
        print("="*70)
        print(f"Analyzing {len(self.sitemap_urls)} English text sitemaps...\n")
        
        # Parse all sitemaps
        results = self.threaded_count_all(max_workers=5)
        
        # Calculate totals
        totals = self.calculate_totals(results)
        
        print("\n" + "="*70)
        print("ANALYSIS RESULTS")
        print("="*70)
        
        # Print per-sitemap breakdown
        print("\n1. PRESS URLS PER SITEMAP:")
        print("-"*40)
        for sitemap_name, count in sorted(totals['press_urls_by_sitemap'].items(), 
                                         key=lambda x: x[1], reverse=True):
            print(f"  {sitemap_name:30} {count:4d} press URLs")
        
        # Print totals
        print("\n2. TOTALS:")
        print("-"*40)
        print(f"  Successful sitemaps:    {totals['successful_sitemaps']}/13")
        print(f"  Failed sitemaps:        {totals['failed_sitemaps']}/13")
        print(f"  Total press URLs (raw): {totals['total_press_urls_raw']}")
        print(f"  Unique press URLs:      {totals['total_press_urls_unique']}")
        print(f"  Average per sitemap:    {totals['avg_press_per_sitemap']:.1f}")
        
        # Calculate scraping time estimates
        if totals['total_press_urls_unique'] > 0:
            print("\n3. SCRAPING TIME ESTIMATES (at 1.5s delay):")
            print("-"*40)
            
            time_estimates = self.estimate_scraping_time(
                totals['total_press_urls_unique'], 
                delay_per_url=1.5
            )
            
            print(f"  For 100 URLs:    {time_estimates['per_100_urls']:.1f} minutes")
            print(f"  For 500 URLs:    {time_estimates['per_500_urls']:.1f} hours")
            print(f"\n  For ALL {totals['total_press_urls_unique']} URLs:")
            print(f"    • {time_estimates['total']['minutes']:.1f} minutes")
            print(f"    • {time_estimates['total']['hours']:.2f} hours")
            print(f"    • {time_estimates['total']['days']:.2f} days")
            
            # Different delay scenarios
            print("\n4. TIME WITH DIFFERENT DELAYS:")
            print("-"*40)
            
            for delay in [1.0, 1.5, 2.0, 2.5]:
                est = self.estimate_scraping_time(
                    totals['total_press_urls_unique'], 
                    delay_per_url=delay
                )
                print(f"  {delay}s delay: {est['total']['hours']:.2f} hours")
        
        # Save results
        self.save_results(results, totals)
        
        return totals
    
    def save_results(self, results, totals):
        """Save analysis results to files"""
        import json
        
        # Save detailed results
        with open('sitemap_analysis_detailed.json', 'w') as f:
            json.dump({
                'sitemap_results': results,
                'totals': totals,
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        # Save just the URLs
        with open('all_press_urls_full.txt', 'w') as f:
            for url in totals['unique_press_urls']:
                f.write(url + '\n')
        
        # Save summary report
        with open('sitemap_analysis_summary.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("BMW PRESS SITEMAP URL ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Sitemaps Analyzed: 13\n")
            f.write(f"Successful Sitemaps: {totals['successful_sitemaps']}\n")
            f.write(f"Failed Sitemaps: {totals['failed_sitemaps']}\n")
            f.write(f"Total Press URLs (raw): {totals['total_press_urls_raw']}\n")
            f.write(f"Unique Press URLs: {totals['total_press_urls_unique']}\n")
            f.write(f"Average URLs per Sitemap: {totals['avg_press_per_sitemap']:.1f}\n\n")
            
            f.write("URLs per Sitemap:\n")
            f.write("-"*40 + "\n")
            for sitemap_name, count in sorted(totals['press_urls_by_sitemap'].items(), 
                                            key=lambda x: x[1], reverse=True):
                f.write(f"{sitemap_name:30} {count:4d}\n")
        
        print(f"\n✓ Results saved to:")
        print(f"  - sitemap_analysis_detailed.json")
        print(f"  - sitemap_analysis_summary.txt")
        print(f"  - all_press_urls_full.txt ({len(totals['unique_press_urls'])} URLs)")

# Run the analysis
if __name__ == "__main__":
    counter = SitemapURLCounter()
    
    try:
        totals = counter.run_analysis()
        
        # Final recommendation
        print("\n" + "="*70)
        print("RECOMMENDED SCRAPING STRATEGY")
        print("="*70)
        
        unique_urls = totals['total_press_urls_unique']
        
        if unique_urls > 0:
            print(f"\nYou have {unique_urls} unique press releases to scrape.")
            print("\nRecommended approaches:")
            print("1. TEST RUN (30 minutes):")
            print("   • Scrape first 100 URLs")
            print("   • Verify quality and selectors")
            
            print("\n2. MEDIUM RUN (2-3 hours):")
            print("   • Scrape 400-500 URLs")
            print("   • Good for initial model training")
            
            print("\n3. FULL RUN:")
            print(f"   • Scrape all {unique_urls} URLs")
            print(f"   • Estimated: {self.estimate_scraping_time(unique_urls, 1.5)['total']['hours']:.1f} hours at 1.5s delay")
            
            print("\nSuggested pipeline configuration:")
            print(f"  max_releases = {min(500, unique_urls)}  # Start with manageable batch")
            print(f"  delay = 1.2  # Slightly faster but still respectful")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()