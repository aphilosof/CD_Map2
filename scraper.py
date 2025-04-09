from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os
import json
import re
import argparse
import requests
from urllib.parse import urljoin

def setup_driver(headless=True, chrome_path=None):
    """Set up Selenium WebDriver with specific Chrome options"""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # If chrome_path is provided, set the binary location
    if chrome_path:
        chrome_options.binary_location = chrome_path
    
    try:
        # Try to create the driver with automatic driver management
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome with WebDriver Manager: {e}")
        print("Trying alternative setup method...")
        
        try:
            # Try with just the options
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("Trying to use Firefox as a fallback...")
            
            try:
                # Try Firefox as a fallback
                from selenium.webdriver.firefox.options import Options as FirefoxOptions
                from selenium.webdriver.firefox.service import Service as FirefoxService
                from webdriver_manager.firefox import GeckoDriverManager
                
                firefox_options = FirefoxOptions()
                if headless:
                    firefox_options.add_argument("--headless")
                    
                driver = webdriver.Firefox(
                    service=FirefoxService(GeckoDriverManager().install()),
                    options=firefox_options
                )
                return driver
            except Exception as e3:
                print(f"Firefox fallback also failed: {e3}")
                print("\nPlease try these solutions:")
                print("1. Install Chrome browser on your system")
                print("2. Provide the path to Chrome with --chrome-path")
                print("3. Install Firefox as a fallback")
                raise Exception("Could not initialize any WebDriver")

def extract_content_from_html(html, author=None):
    """Extract article content between specific markers"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the post content
    content = soup.select_one('div.post-content')
    if not content:
        return None, None
    
    # Get title
    title_elem = soup.find('h1')
    title = title_elem.text.strip() if title_elem else "Untitled Article"
    
    # Get the text
    full_text = content.get_text()
    full_text = re.sub(r'\n+', '\n', full_text)
    full_text = re.sub(r' +', ' ', full_text)
    full_text = full_text.strip()
    
    # Extract between markers
    start_marker = None
    end_marker = "Subscribe to Climate Drift"
    
    # Check if it's by Skander Garroum
    if author and "Skander Garroum" in author:
        start_marker = "Apply now"
    else:
        # Try to find "But first, who is...?" pattern
        match = re.search(r'But first, who is [^?]+\?', full_text)
        if match:
            start_marker = match.group(0)
    
    # Extract content
    if start_marker and start_marker in full_text:
        start_idx = full_text.index(start_marker) + len(start_marker)
        article_text = full_text[start_idx:]
    else:
        article_text = full_text
    
    if end_marker in article_text:
        end_idx = article_text.index(end_marker)
        article_text = article_text[:end_idx]
    
    article_text = article_text.strip()
    
    return title, article_text

def get_all_article_links(driver, base_url):
    """Get all article links from Climate Drift using Selenium"""
    article_links = []
    
    # Collect archive URLs (may be paginated or by year)
    archive_urls = [f"{base_url}/archive"]
    
    # Add year-specific archive pages (common for Substack)
    current_year = 2025  # Current year as of script creation
    for year in range(2020, current_year + 1):
        archive_urls.append(f"{base_url}/archive?year={year}")
    
    # Process each archive page
    for archive_url in archive_urls:
        try:
            print(f"Loading archive page: {archive_url}")
            driver.get(archive_url)
            
            # Wait for content to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".post-preview, .archive-post, article, .post"))
                )
            except TimeoutException:
                print(f"  Timeout waiting for content to load on {archive_url}")
                continue
            
            # Scroll down a few times to load more content
            for i in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                print(f"  Scrolling ({i+1}/5)...")
                time.sleep(2)
            
            # Get links after scrolling
            post_links = driver.find_elements(By.TAG_NAME, "a")
            page_links = []
            for link in post_links:
                try:
                    href = link.get_attribute("href")
                    if href and "/p/" in href:
                        # Clean URL if it has /comments
                        if href.endswith("/comments"):
                            href = href[:-9]
                        page_links.append(href)
                except:
                    pass  # Skip any links that cause errors
            
            article_links.extend(page_links)
            print(f"  Found {len(page_links)} articles on this page")
            
        except Exception as e:
            print(f"Error processing {archive_url}: {e}")
    
    # Also check homepage
    try:
        print(f"Loading homepage: {base_url}")
        driver.get(base_url)
        
        # Wait for content to load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except TimeoutException:
            print("  Timeout waiting for homepage content")
        
        # Get links
        post_links = driver.find_elements(By.TAG_NAME, "a")
        home_links = []
        for link in post_links:
            try:
                href = link.get_attribute("href")
                if href and "/p/" in href:
                    # Clean URL if it has /comments
                    if href.endswith("/comments"):
                        href = href[:-9]
                    home_links.append(href)
            except:
                pass  # Skip any links that cause errors
        
        article_links.extend(home_links)
        print(f"  Found {len(home_links)} articles on homepage")
    except Exception as e:
        print(f"Error processing homepage: {e}")
    
    # Remove duplicates
    article_links = list(set(article_links))
    
    # Ensure all links are post links
    filtered_links = [link for link in article_links if "/p/" in link]
    
    print(f"Found {len(filtered_links)} unique article links")
    return filtered_links

def download_article_content(url, headers):
    """Download article content using requests (non-Selenium approach)"""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        return None
    except Exception as e:
        print(f"  Error downloading with requests: {e}")
        return None

def save_text_file(content, filename, directory):
    """Save text to a file"""
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Extract text from Climate Drift articles using Selenium")
    parser.add_argument("--output", "-o", default="./climate_drift_text", help="Output directory")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window while scraping")
    parser.add_argument("--chrome-path", help="Path to Chrome executable (if not in standard location)")
    parser.add_argument("--urls-only", action="store_true", help="Only extract URLs, don't download content")
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug directory
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    base_url = "https://www.climatedrift.com"
    
    # Setup Selenium WebDriver
    try:
        driver = setup_driver(headless=not args.no_headless, chrome_path=args.chrome_path)
    except Exception as e:
        print(f"Failed to initialize WebDriver: {e}")
        return
    
    try:
        # Get all article links
        article_links = get_all_article_links(driver, base_url)
        
        # Save list of URLs
        with open(os.path.join(output_dir, "article_urls.txt"), 'w', encoding='utf-8') as f:
            for url in article_links:
                f.write(f"{url}\n")
        
        # Save as JSON too
        with open(os.path.join(debug_dir, "article_links.json"), 'w', encoding='utf-8') as f:
            json.dump(article_links, f, indent=2)
        
        print(f"Saved {len(article_links)} article URLs to {os.path.join(output_dir, 'article_urls.txt')}")
        
        # Skip content download if only URLs are requested
        if args.urls_only:
            print("URLs extraction complete. Skipping content download as requested.")
            driver.quit()
            return
        
        # Process each article
        successful_articles = []
        failed_urls = []
        
        # Headers for fallback requests method
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        for i, url in enumerate(article_links):
            print(f"Processing article {i+1}/{len(article_links)}: {url}")
            
            try:
                # Try Selenium approach first
                try:
                    driver.get(url)
                    
                    # Wait for content to load
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.post-content, article"))
                    )
                    
                    # Get page HTML
                    html = driver.page_source
                except Exception as selenium_err:
                    print(f"  Selenium error: {selenium_err}")
                    print("  Trying fallback with requests...")
                    html = download_article_content(url, headers)
                    
                    if not html:
                        print("  Fallback also failed, skipping article")
                        failed_urls.append(url)
                        continue
                
                # Save raw HTML for debugging (first few articles)
                if i < 5:
                    with open(os.path.join(debug_dir, f"article_{i+1}.html"), 'w', encoding='utf-8') as f:
                        f.write(html)
                
                # Extract author - try both methods
                author = None
                soup = BeautifulSoup(html, 'html.parser')
                author_elem = soup.select_one(".byline, .author-name, .substack-author")
                if author_elem:
                    author = author_elem.text.strip()
                
                # Extract content
                title, article_text = extract_content_from_html(html, author)
                
                if not title or not article_text or len(article_text) < 100:
                    print("  Skipping (no content extracted)")
                    failed_urls.append(url)
                    continue
                
                # Create safe filename
                safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
                safe_title = safe_title[:50]
                
                # Save article
                filename = f"{i+1:03d}_{safe_title}.txt"
                filepath = save_text_file(article_text, filename, output_dir)
                
                print(f"  Saved: {filename} ({len(article_text)} chars)")
                
                # Add to successful articles
                successful_articles.append({
                    "title": title,
                    "url": url,
                    "author": author,
                    "filename": filename,
                    "chars": len(article_text)
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                failed_urls.append(url)
            
            # Be respectful with rate limiting
            time.sleep(2)
        
        # Save metadata
        with open(os.path.join(output_dir, "articles_info.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "source": base_url,
                "articles_count": len(successful_articles),
                "articles": successful_articles
            }, f, indent=2)
        
        # Save failed URLs
        if failed_urls:
            with open(os.path.join(debug_dir, "failed_urls.txt"), 'w', encoding='utf-8') as f:
                for url in failed_urls:
                    f.write(f"{url}\n")
        
        print(f"\nExtraction complete!")
        print(f"Saved {len(successful_articles)} articles to {output_dir}")
        print(f"Failed to process {len(failed_urls)} URLs")
        
    finally:
        # Clean up
        driver.quit()

if __name__ == "__main__":
    main()