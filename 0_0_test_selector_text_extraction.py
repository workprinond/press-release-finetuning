import requests
from bs4 import BeautifulSoup

# Your target URL
url = "https://www.press.bmwgroup.com/pressclub/p/pcgl/pressDetail.html?title=bmw-m-racing-academy-class-of-2026-starts-with-three-customer-racing-drivers-from-three-continents&outputChannelId=6&id=T0454343EN&left_menu_item=node__10883"

# Your specific selectors (keeping the same selector that works)
HEADLINE_SELECTOR = "#content > div > div.content-left > div > h1"
SUBHEADLINE_SELECTOR = "#content > div > div.content-left > div > h2"
AUTHOR_SELECTOR = "#content > div > div.content-left > div > div.left > p:nth-child(4)"  
ARTICLE_BODY_SELECTOR = "#article-text > p"

def clean_text(element):
    """Helper function to clean text, removing ALL formatting tags"""
    if not element:
        return None
    
    # Create a copy to avoid modifying the original soup
    element_copy = BeautifulSoup(str(element), 'html.parser')
    
    # Remove specific formatting tags completely
    for tag in element_copy.find_all(['strong', 'b', 'em', 'i', 'u', 'span']):
        if tag and hasattr(tag, 'unwrap'):
            tag.unwrap()
    
    # Replace <br> tags with a space for better text flow
    for br in element_copy.find_all("br"):
        if br:
            br.replace_with(" ")
    
    # Get text and clean up special spaces and excessive whitespace
    text = element_copy.get_text(separator=" ", strip=True)
    if text:
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        text = ' '.join(text.split())
    
    return text

try:
    print(f"Fetching: {url[:70]}...")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    print(f"Status: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # DEBUG: Check if we can find the main containers
    print("\nDEBUG - Checking key elements:")
    print(f"- 'content' div: {'FOUND' if soup.select_one('#content') else 'NOT FOUND'}")
    print(f"- 'article-text' div: {'FOUND' if soup.select_one('#article-text') else 'NOT FOUND'}")
    
    # Extract elements using your selectors
    headline_elem = soup.select_one(HEADLINE_SELECTOR)
    subheadline_elem = soup.select_one(SUBHEADLINE_SELECTOR)
    author_elem = soup.select_one(AUTHOR_SELECTOR)  # Correctly named as author element
    article_body_elems = soup.select(ARTICLE_BODY_SELECTOR) if soup.select_one('#article-text') else []
    
    print(f"\nDEBUG - Elements found:")
    print(f"Headline element: {'FOUND' if headline_elem else 'NOT FOUND'}")
    print(f"Subheadline element: {'FOUND' if subheadline_elem else 'NOT FOUND'}")
    print(f"Author element: {'FOUND' if author_elem else 'NOT FOUND'}")
    print(f"Article body paragraphs: {len(article_body_elems)} found")
    
    # Clean the extracted text (only if elements exist)
    headline = clean_text(headline_elem) if headline_elem else None
    subheadline = clean_text(subheadline_elem) if subheadline_elem else None
    author_info = clean_text(author_elem) if author_elem else None  # Correct variable name
    
    # Combine all article body paragraphs
    article_body = ""
    if article_body_elems:
        for i, p in enumerate(article_body_elems):
            cleaned_p = clean_text(p)
            if cleaned_p:
                article_body += cleaned_p + "\n\n"
    else:
        print("\nWARNING: No article body paragraphs found!")
        alt_elems = soup.select('.textModule p, .article-body p, p')
        print(f"Trying alternative selectors: {len(alt_elems)} paragraphs found")
        for i, p in enumerate(alt_elems[:10]):
            cleaned_p = clean_text(p)
            if cleaned_p:
                article_body += cleaned_p + "\n\n"
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    print(f"1. HEADLINE:")
    print(f"   {headline if headline else 'NOT FOUND'}")
    
    print(f"\n2. SUBHEADLINE:")
    print(f"   {subheadline if subheadline else 'NOT FOUND'}")
    
    print(f"\n3. AUTHOR:")  # Correct label
    print(f"   {author_info if author_info else 'NOT FOUND'}")
    
    print(f"\n4. ARTICLE BODY:")
    if article_body:
        print(f"   Total characters: {len(article_body)}")
        print(f"   Total paragraphs: {len(article_body_elems)}")
        print(f"\n   SAMPLE (first 1000 chars):")
        print(f"   {article_body[:1000]}...")
        
        with open('article_output.txt', 'w', encoding='utf-8') as f:
            f.write(article_body)
        print(f"\n   Full article saved to 'article_output.txt'")
        
        # Also save structured data with metadata
        structured_data = {
            'headline': headline,
            'subheadline': subheadline,
            'author': author_info,
            'article_body': article_body,
            'url': url,
            'total_paragraphs': len(article_body_elems),
            'total_characters': len(article_body)
        }
        
        with open('structured_article.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        print(f"   Structured data saved to 'structured_article.json'")
    else:
        print("   NO CONTENT FOUND")
        
    # Save raw HTML for debugging
    with open('debug_page.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))
    print(f"\nDEBUG: Raw HTML saved to 'debug_page.html'")
        
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()