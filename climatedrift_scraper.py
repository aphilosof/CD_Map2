import requests
from bs4 import BeautifulSoup
import os
import re
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager # Automatically manages chromedriver

# --- Configuration ---
ARCHIVE_URL = "https://www.climatedrift.com/archive"
OUTPUT_DIR = "climatedrift_articles_selenium" # Changed output dir name slightly
BASE_URL = "https://www.climatedrift.com"

# Headers to mimic a browser request (used for individual article fetching)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Markers for text extraction (same as before - PLEASE VERIFY THESE)
DEFAULT_START_REGEX = r"But first, who is .*?\?"
SKANDER_AUTHOR_NAME = "Skander Garroum"
SKANDER_START_MARKER = "Let’s Apply" # Text *after* which to start (Verify this!)
END_MARKER = "Subscribe to Climate Drift" # Text *before* which to stop

# Delays (seconds)
REQUEST_DELAY = 1.5 # Delay between fetching individual articles
SCROLL_PAUSE_TIME = 3 # How long to wait after scrolling down the archive page
INITIAL_LOAD_WAIT = 5 # How long to wait for initial archive page load

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def setup_selenium_driver():
    """Configures and returns a Selenium WebDriver instance."""
    logging.info("Setting up Selenium WebDriver...")
    options = Options()
    # options.add_argument("--headless")  # Uncomment to run Chrome without opening a visible window
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox") # Often needed for running in specific environments
    options.add_argument("--window-size=1920,1080") # Specify window size
    options.add_argument(f"user-agent={HEADERS['User-Agent']}") # Set user agent
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Suppress certain logs

    try:
        # Use webdriver-manager to handle driver installation/updates
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        logging.info("Selenium WebDriver setup complete.")
        return driver
    except Exception as e:
        logging.error(f"Failed to setup Selenium WebDriver: {e}", exc_info=True)
        logging.error("Ensure Google Chrome is installed and webdriver-manager can download the driver.")
        return None


def fetch_html_requests(url):
    """Fetches HTML content using requests (for individual articles)."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        logging.debug(f"Successfully fetched with requests: {url}")
        return response.text
    except requests.exceptions.Timeout:
        logging.error(f"Timeout occurred while fetching {url} with requests")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url} with requests: {e}")
    return None

def sanitize_filename(title):
    """Removes invalid characters for filenames and limits length."""
    if not title:
        title = "untitled_article"
    sanitized = re.sub(r'[^\w\-\s]+', '', title) # Allow word chars, hyphen, space
    sanitized = re.sub(r'\s+', '_', sanitized).strip('_') # Replace space with underscore
    sanitized = re.sub(r'_+', '_', sanitized) # Collapse multiple underscores
    return sanitized[:150] # Limit length

def extract_article_data(html_content, article_url):
    """Extracts author, title, and relevant text from article HTML (same logic as before)."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # --- Find Author --- (Verify selector if needed)
    author_tag = soup.select_one('a.pencraft[href*="/u/"]') # Example selector
    author_name = author_tag.get_text(strip=True) if author_tag else "Unknown Author"
    logging.info(f"Author identified: {author_name}")

    # --- Find Title --- (Verify selector if needed)
    title_tag = soup.find('h1', class_=lambda x: x and 'post-title' in x) # More specific title selector
    if not title_tag:
        title_tag = soup.find('h1') # Fallback
    title = title_tag.get_text(strip=True) if title_tag else f"Untitled_{article_url.split('/')[-1]}"

    # --- Find Article Body --- (Verify selector if needed)
    article_body_tag = soup.select_one('div.available-content .body.markup') # More specific body selector
    if not article_body_tag:
         article_body_tag = soup.select_one('div.available-content') # Fallback
    if not article_body_tag:
        logging.warning(f"Could not find article body container for {article_url}")
        return None, title, author_name

    article_text = article_body_tag.get_text(separator='\n', strip=True)

    # --- Determine Start and End Points --- (Same logic as before)
    start_index = 0
    end_index = len(article_text)

    end_marker_pos = article_text.find(END_MARKER)
    if end_marker_pos != -1:
        end_index = end_marker_pos
        logging.info(f"Found end marker '{END_MARKER}'.")
    else:
        logging.warning(f"End marker '{END_MARKER}' not found in {article_url}. Including text until end.")

    if author_name.strip().lower() == SKANDER_AUTHOR_NAME.lower():
        start_marker_pos = article_text.lower().find(SKANDER_START_MARKER.lower())
        if start_marker_pos != -1:
            start_index = start_marker_pos + len(SKANDER_START_MARKER)
            logging.info(f"Found start marker '{SKANDER_START_MARKER}' for author {author_name}.")
        else:
            logging.warning(f"Start marker '{SKANDER_START_MARKER}' not found for author {author_name} in {article_url}. Starting from beginning.")
            start_index = 0
    else:
        start_match = re.search(DEFAULT_START_REGEX, article_text, re.IGNORECASE | re.DOTALL)
        if start_match:
            start_index = start_match.end()
            logging.info("Found start marker 'But first, who is...?'.")
        else:
            logging.warning(f"Default start marker regex not found in {article_url}. Starting from beginning.")
            start_index = 0

    if start_index >= end_index:
         logging.warning(f"Start marker found after or at the end marker position in {article_url}. Extracting empty text.")
         extracted_text = ""
    else:
        extracted_text = article_text[start_index:end_index].strip()

    return extracted_text, title, author_name

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Create output directory
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            logging.info(f"Created directory: {OUTPUT_DIR}")
        except OSError as e:
            logging.error(f"Failed to create directory {OUTPUT_DIR}: {e}")
            exit()

    # 2. Use Selenium to get the fully loaded archive page HTML
    driver = setup_selenium_driver()
    archive_html = None
    article_urls = []

    if driver:
        try:
            logging.info(f"Navigating to archive page with Selenium: {ARCHIVE_URL}")
            driver.get(ARCHIVE_URL)
            logging.info(f"Waiting {INITIAL_LOAD_WAIT} seconds for initial page load...")
            time.sleep(INITIAL_LOAD_WAIT)

            # Scroll down the page to trigger dynamic loading
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 30 # Increased max scrolls
            logging.info("Starting to scroll down to load all articles...")

            while scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                logging.info(f"Scrolled down. Waiting {SCROLL_PAUSE_TIME} seconds...")
                time.sleep(SCROLL_PAUSE_TIME)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logging.info("Reached end of scrollable content.")
                    break
                last_height = new_height
                scroll_attempts += 1
                logging.info(f"Scroll attempt {scroll_attempts} complete. New height: {new_height}")

            if scroll_attempts == max_scroll_attempts:
                logging.warning(f"Reached maximum scroll attempts ({max_scroll_attempts}). May not have loaded all articles.")

            # Get the final page source after scrolling
            logging.info("Retrieving final page source...")
            archive_html = driver.page_source
            logging.info("Successfully retrieved full page source using Selenium.")

        except Exception as e:
            logging.error(f"An error occurred during Selenium operation: {e}", exc_info=True)
        finally:
            logging.info("Closing Selenium WebDriver.")
            driver.quit()
    else:
        logging.error("Selenium WebDriver could not be initialized. Cannot proceed.")
        exit()


    # 3. Parse the obtained HTML with BeautifulSoup to find article links
    if archive_html:
        logging.info("Parsing the archive page HTML with BeautifulSoup...")
        soup_archive = BeautifulSoup(archive_html, 'html.parser')

        # Adjust selector based on inspection of the fully loaded archive page
        # This looks for links within post previews that point to /p/ pages
        link_tags = soup_archive.select('div.post-preview a.pencraft[href*="/p/"]') # Primary selector attempt
        if not link_tags:
             logging.warning("Primary selector 'div.post-preview a.pencraft[href*=\"/p/\"]' found no links. Trying broader search 'a[href*=\"/p/\"]'.")
             link_tags = soup_archive.select('a[href*="/p/"]') # Broader fallback

        temp_urls = set() # Use a set to automatically handle duplicates
        for tag in link_tags:
            href = tag.get('href')
            if href:
                # Ensure URL is absolute
                if href.startswith('/'):
                    href = BASE_URL + href
                # Filter only post links from the correct domain
                if href.startswith(BASE_URL + "/p/"):
                    temp_urls.add(href)

        article_urls = list(temp_urls) # Convert back to list

        if not article_urls:
            logging.error("Could not find any article links matching the pattern on the loaded archive page. Please check the HTML structure and update the selectors (e.g., 'div.post-preview a.pencraft') in the script.")
        else:
            logging.info(f"Found {len(article_urls)} unique article links after scrolling.")
    else:
        logging.error("Failed to get archive page content using Selenium. No links to process.")


    # 4. Process each article link found
    total_articles = len(article_urls)
    if total_articles > 0:
        logging.info(f"--- Starting to fetch and process {total_articles} individual articles ---")
        for i, article_url in enumerate(article_urls):
            logging.info(f"--- Processing article {i+1}/{total_articles}: {article_url} ---")

            article_html = fetch_html_requests(article_url) # Use requests for individual pages
            if not article_html:
                logging.warning(f"Skipping article due to fetch error: {article_url}")
                time.sleep(REQUEST_DELAY) # Wait even on failure
                continue

            try:
                # Extract data using the existing function
                extracted_text, title, author = extract_article_data(article_html, article_url)

                if extracted_text is None:
                     logging.warning(f"Skipping article due to extraction issue (e.g., body not found): {article_url}")
                elif not extracted_text.strip(): # Check if extracted text is just whitespace
                     logging.warning(f"Extracted text is empty or only whitespace for '{title}' ({article_url}). Check markers/content.")
                else:
                    # 5. Save extracted text to a DEDICATED file
                    filename = os.path.join(OUTPUT_DIR, f"{sanitize_filename(title)}.txt")
                    try:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"Title: {title}\n")
                            f.write(f"Author: {author}\n")
                            f.write(f"URL: {article_url}\n")
                            f.write("="*30 + "\n\n") # Separator
                            f.write(extracted_text.strip()) # Write the cleaned text
                        logging.info(f"Successfully saved: {filename}")
                    except IOError as e:
                        logging.error(f"Could not write file {filename}: {e}")
                    except Exception as e:
                        logging.error(f"An unexpected error occurred while writing {filename}: {e}")

            except Exception as e:
                logging.error(f"An unexpected error occurred during data extraction/saving for {article_url}: {e}", exc_info=True)

            # Be polite - wait before the next request
            logging.debug(f"Waiting for {REQUEST_DELAY} seconds...")
            time.sleep(REQUEST_DELAY)

    logging.info(f"--- Finished processing. {total_articles} article links were found and attempted. Check logs for details. ---")
    logging.info(f"Text files (one per article) should be in the '{OUTPUT_DIR}' directory.")