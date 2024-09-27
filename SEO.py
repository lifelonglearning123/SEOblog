import streamlit as st
import re
from bs4 import BeautifulSoup
from collections import Counter
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import openai
from dotenv import load_dotenv
import os
from simple_password_auth import authenticate_user  # Import the function from the separate file
import json
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import random
from zenrows import ZenRowsClient

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to create a filename from a URL
def create_filename_from_url(url):
    """
    Creates a sanitized filename from a given URL.

    Parameters:
    - url (str): The URL to generate the filename from.

    Returns:
    - str: A sanitized filename.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  # Extract domain (e.g., "example.com")
    # Sanitize domain to remove unwanted characters
    sanitized_domain = re.sub(r'[^a-zA-Z0-9]', '_', domain)  # Replace non-alphanumeric chars with '_'
    return f"{sanitized_domain}_content.json"

# Function to save data to a JSON file
def save_data_to_json(data, filename):
    """
    Saves the given data to a JSON file.

    Parameters:
    - data: The data to save.
    - filename (str): The filename to save the data to.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

# Function to load data from a JSON file
def load_data_from_json(filename="company_brand.json"):
    """
    Loads data from a JSON file.

    Parameters:
    - filename (str): The filename to load the data from.

    Returns:
    - data: The data loaded from the file or None if file not found.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return None  # File not found, return None

# Function to generate a summary (blog content) using OpenAI's GPT-4 model
def summary(full_response, all_results):
    """
    Generates a blog article using OpenAI's GPT-4 model.

    Parameters:
    - full_response (str): The topic for the article.
    - all_results (str): Keywords to include in the article.

    Returns:
    - summary_text (str): The generated article.
    """
    # Prepare the prompt for the model
    prompt = (
        "Write an article on the following topic: " + full_response +
        ". Use the following keywords within the article: " + all_results +
        ". Write a blog in 1500 words. Use a descriptive and engaging title. "
        "Include the target keyword in the title. Use subheadings to break the content. "
        "Include a call to action."
    )

    # Call the OpenAI API to generate the article
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the best blog writer in the world."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000
    )
    summary_text = response.choices[0].message.content.strip()
    return summary_text

# Function to validate a keyword
def is_valid_keyword(keyword):
    """
    Checks if a keyword is valid (not a common stop word and is alphabetic).

    Parameters:
    - keyword (str): The keyword or phrase to validate.

    Returns:
    - bool: True if valid, False otherwise.
    """
    common_words = {'can', 'an', 'at', 'but', 'the', 'and', 'a', 'is', 'in', 'it', 'of', 'to', 'for', 'on', 'with', 'as', 'by', 'that', 'be', 'am', 'are'}
    # Split the phrase into individual words
    words = keyword.split()
    # Check if each word is alphabetic and not a common word
    valid_words = [word for word in words if word.isalpha() and word.lower() not in common_words]
    # The phrase is valid if it has the same number of valid words as the original
    return len(valid_words) == len(words)

# Function to fetch webpage content, using Selenium if necessary
def get_webpage_content(url, session=None):
    """
    Fetches the content of a webpage, automatically determining whether to use Selenium for JavaScript-rendered pages.

    Parameters:
    - url (str): The URL of the webpage to fetch.
    - session (requests.Session): An optional session object to handle cookies and sessions.

    Returns:
    - content (str): The HTML content of the webpage.
    """
    try:
        # Implement random delay between requests to mimic human behavior
        time.sleep(random.uniform(1, 5))

        # Headers to mimic a regular browser visit
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/93.0.4577.63 Safari/537.36'),
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
        }

        client = ZenRowsClient("ff2968811cfac24356bb7e0bcfdc434ca99951d2")
        response = client.get(url)
        print("This is benefiz response", response)
        # Check for 403 status code
        if response.status_code == 403:
            print("Received 403 status code. Switching to Selenium.")
            content = get_content_with_selenium(url, headers)
        else:
            response.raise_for_status()
            content = response.text

        # Check if content is likely incomplete or empty
        if is_content_valid(content):
            return content
        else:
            print("Content seems incomplete. Using Selenium for JavaScript-rendered content.")
            content = get_content_with_selenium(url, headers)
            return content

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        # Try with Selenium
        print("Trying with Selenium due to HTTP error.")
        content = get_content_with_selenium(url, headers)
        return content
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        # Try with Selenium
        print("Trying with Selenium due to Request error.")
        content = get_content_with_selenium(url, headers)
        return content
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Function to check if content is valid
def is_content_valid(content):
    """
    Checks if the fetched content is valid or likely requires JavaScript rendering.

    Parameters:
    - content (str): The HTML content of the webpage.

    Returns:
    - bool: True if content is valid, False if content is likely incomplete.
    """
    # Simple heuristic: Check if certain tags or keywords are present
    soup = BeautifulSoup(content, 'html.parser')
    # Check if there's meaningful content in the <body>
    body_text = soup.body.get_text(strip=True) if soup.body else ''
    if len(body_text) > 200:  # Threshold for meaningful content
        return True
    else:
        return False

# Function to fetch content using Selenium
def get_content_with_selenium(url, headers):
    """
    Fetches the content of a webpage using Selenium to render JavaScript.

    Parameters:
    - url (str): The URL of the webpage to fetch.
    - headers (dict): A dictionary of HTTP headers.

    Returns:
    - content (str): The HTML content of the webpage.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    # Implement random delay to mimic human behavior
    time.sleep(random.uniform(1, 3))

    options = Options()
    options.add_argument('--headless')  # Comment this line to see the browser
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'user-agent={headers["User-Agent"]}')
    options.add_argument('--window-size=1920,1080')

    # Use webdriver_manager to handle ChromeDriver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)

        # Wait for the body element to be loaded
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )

        # Scroll to the bottom to trigger dynamic content loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for the content to load

        content = driver.page_source

    except Exception as e:
        print(f"An error occurred while using Selenium: {e}")
        content = None
    finally:
        driver.quit()

    return content

# Function to get the first n words of text
def get_first_n_words(text, n=50):
    """
    Returns the first n words of a text.

    Parameters:
    - text (str): The text to process.
    - n (int): The number of words to return.

    Returns:
    - str: The first n words of the text.
    """
    words = text.split()[:n]
    return ' '.join(words) + '...'

# Function to clean text by removing stopwords and punctuation
def clean_text(text):
    """
    Cleans text by lowercasing, removing punctuation, and removing stopwords.

    Parameters:
    - text (str): The text to clean.

    Returns:
    - list: A list of cleaned words.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# Function to get high-frequency words
def get_high_frequency_words(text, num=10):
    """
    Returns the top num high-frequency words in the text.

    Parameters:
    - text (str): The text to analyze.
    - num (int): The number of top words to return.

    Returns:
    - list: A list of tuples (word, count).
    """
    words = clean_text(text)
    words = [word for word in words if is_valid_keyword(word)]
    counter = Counter(words)
    most_common = counter.most_common(num)
    return most_common

# Function to get high-frequency phrases
def get_high_frequency_phrases(text, num=10):
    """
    Returns the top num high-frequency two-word phrases in the text.

    Parameters:
    - text (str): The text to analyze.
    - num (int): The number of top phrases to return.

    Returns:
    - list: A list of tuples (phrase, count).
    """
    words = clean_text(text)
    # Construct candidate phrases from adjacent cleaned words
    phrases = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    # Filter out any non-valid phrases
    valid_phrases = [phrase for phrase in phrases if is_valid_keyword(phrase)]
    counter = Counter(valid_phrases)
    most_common = counter.most_common(num)
    return most_common

# Function to extract LSI keywords
def get_lsi_keywords(text, num=10):
    """
    Extracts LSI keywords from the given text.

    Parameters:
    - text (str): The text to analyze.
    - num (int): The number of keywords to return.

    Returns:
    - list: A list of keywords.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    # Preprocess the text
    def preprocess_text(text):
        # Remove HTML tags if any
        text = BeautifulSoup(text, 'html.parser').get_text()
        # Convert to lowercase
        text = text.lower()
        # Remove numbers and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    processed_text = preprocess_text(text)

    # Split text into sentences for documents
    documents = re.split(r'(?<=[.!?])\s+', processed_text)
    documents = [doc.strip() for doc in documents if doc.strip()]
    if not documents:
        raise ValueError("No valid documents to process after preprocessing.")

    # Initialize TfidfVectorizer with appropriate parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2)
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        raise ValueError(f"Error in vectorization: {e}")

    if tfidf_matrix.shape[1] == 0:
        raise ValueError("TF-IDF matrix is empty. Check your documents and preprocessing steps.")

    # Perform LSI using TruncatedSVD
    svd = TruncatedSVD(n_components=min(100, tfidf_matrix.shape[1]-1), random_state=42)
    lsi = make_pipeline(svd, Normalizer(copy=False))

    lsi_matrix = lsi.fit_transform(tfidf_matrix)

    # Get terms and their scores
    terms = vectorizer.get_feature_names_out()
    total_scores = svd.components_.sum(axis=0)
    term_scores = list(zip(terms, total_scores))

    # Sort terms by score in descending order
    sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)

    # Filter valid keywords and remove duplicates
    keywords = []
    for term, score in sorted_terms:
        if is_valid_keyword(term) and term not in keywords:
            keywords.append(term)
        if len(keywords) >= num:
            break

    return keywords

# Function to provide best practices for a blog article
def best_practices_for_blog(article_length):
    """
    Returns best practices for a blog article based on its length.

    Parameters:
    - article_length (int): The length of the article in words.

    Returns:
    - dict: A dictionary containing counts of headings, images, and tips.
    """
    h1_count = 1
    h2_count = max(2, article_length // 500)
    h3_count = max(2, article_length // 250)
    h4_count = max(2, article_length // 125)
    image_count = max(2, article_length // 250)
    tips = [
        "Use descriptive and engaging title.",
        "Include target keyword in the title.",
        "Use subheadings to break the content.",
        "Include internal and external links.",
        "Optimize images with alt text containing target keyword.",
        "Use bullet points or numbered lists.",
        "Include a call to action.",
        "Keep paragraphs short and easy to read.",
        "Include social share buttons."
    ]
    return {
        "h1_count": h1_count,
        "h2_count": h2_count,
        "h3_count": h3_count,
        "h4_count": h4_count,
        "image_count": image_count,
        "tips": tips
    }

# Function to extract text from HTML content
def extract_text_from_html(html_content):
    """
    Extracts meaningful text from HTML content.

    Parameters:
    - html_content (str): The HTML content to extract text from.

    Returns:
    - str: The extracted text.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove unwanted tags
    for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
        script_or_style.decompose()
    # Extract text
    text = soup.get_text(separator=' ', strip=True)
    # Clean up the text
    text = ' '.join(text.split())
    return text

# Streamlit UI function
def streamlit_ui():
    """
    Runs the Streamlit UI for the SEO Content Writer application.
    """
    st.title("SEO Content Writer")
    all_results = ""
    article_length = 1500  # Set default article length

    # For now, we are examining one URL
    num_urls = 1
    urls = [st.text_input(f"Please enter webpage URL for keyword analysis {i+1}:", key=f"url_{i}") for i in range(num_urls)]
    article_topic = st.text_input("Enter the topic for your article:")

    if st.button("Generate"):
        for i, url in enumerate(urls):
            if url:  # Ensure URL is not empty
                html_content = get_webpage_content(url)
                if html_content is None:
                    st.write("Failed to retrieve content, please check the URL and try again.")
                    continue  # Skip further processing for this URL
                text = extract_text_from_html(html_content)
                if len(text) < 10:
                    st.write("Extracted text is too short. Trying alternative extraction method.")
                    text = extract_text_from_html(html_content)

                # Proceed only if text is not empty
                if len(text) < 10:
                    st.write("Failed to extract sufficient content from the page.")
                    continue

                try:
                    lsi_keywords = get_lsi_keywords(text)
                except Exception as e:
                    print(f"Error extracting LSI keywords: {e}")
                    lsi_keywords = []

                high_freq_keywords = get_high_frequency_words(text)
                high_freq_phrases = get_high_frequency_phrases(text)

                # Format the keywords and phrases
                keywords_string = ", ".join([f"{kw[0]} ({kw[1]})" for kw in high_freq_keywords])
                phrases_string = ", ".join([f"{phrase[0]} ({phrase[1]})" for phrase in high_freq_phrases])

                all_results += f"{phrases_string} "

                st.write(f"Analysis for URL: {url}")
                st.write(f"Text Preview: {get_first_n_words(text)}")
                st.write("LSI Keywords:", ", ".join(lsi_keywords))
                st.write("High Frequency Keywords:", keywords_string)
                st.write("High Frequency Two-Word Phrases:", phrases_string)
                st.write("=" * 50)

        # Generate summary with OpenAI GPT
        if all_results:
            with st.spinner("Generating blog content..."):
                blog_content = summary(article_topic, all_results)

                # Create a dynamic filename based on the URL
                dynamic_filename = create_filename_from_url(url)

                # Save the blog content with a dynamic filename
                save_data_to_json(blog_content, filename=dynamic_filename)

                st.subheader("Generated Blog Content")
                st.write(blog_content)
        else:
            st.write("Please enter a topic for the article and ensure analysis was successful.")

# Main block to run the app
if __name__ == "__main__":
    if authenticate_user():  # If authenticated successfully, run the app
        streamlit_ui()
