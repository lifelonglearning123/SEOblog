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
import re
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import random



load_dotenv()  # This loads the environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")


def create_filename_from_url(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  # Extract domain (e.g., "example.com")
    # Sanitize domain to remove unwanted characters
    sanitized_domain = re.sub(r'[^a-zA-Z0-9]', '_', domain)  # Replace non-alphanumeric chars with '_'
    return f"{sanitized_domain}_content.json"


#Declare functions to allow loading of JSON file
def save_data_to_json(data, filename):
    """Save the given data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_data_from_json(filename="company_brand.json"):
    """Load data from a JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return None  # File not found, return None

def summary(full_response, all_results):
    #print("**************")
    full_response = "write articles on the following topic: " + full_response + "Use following keywords within the articles " + all_results + "Write a blog in 1500 words. Use descriptive and engaging title. Include target keyword in the title. Use subheadings to break the content. Include a call to action."
    #print(full_response)
    response = openai.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {   "role": "system",
            "content" : "You are the best blog writer in the world. Write a blog over 1500 words ",
                "role":"user",
            "content": full_response
            }
        ],
        max_tokens=4000
    )
    summary_text = response.choices[0].message.content
    return summary_text

def is_valid_keyword(keyword):
    common_words = {'can', 'an', 'at', 'but', 'the', 'and', 'a', 'is', 'in', 'it', 'of', 'to', 'for', 'on', 'with', 'as', 'by', 'that', 'be', 'am', 'are'}
    # Split the phrase into individual words
    words = keyword.split()
    # Check if each word is alphabetic and not a common word
    valid_words = [word for word in words if word.isalpha() and word.lower() not in common_words]
    # The phrase is valid if it has the same number of valid words as the original
    return len(valid_words) == len(words)


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

        # Use requests with a session to handle cookies and sessions
        if session is None:
            session = requests.Session()
        session.headers.update(headers)
        response = session.get(url)

        # Check for 403 status code
        if response.status_code == 403:
            print("Received 403 status code. Switching to Selenium.")
            content = get_content_with_selenium(url, headers)
            print("Stage check 403")
        else:
            response.raise_for_status()
            content = response.text

        # Check if content is likely incomplete or empty
        if is_content_valid(content):
            return content
        else:
            print("Content seems incomplete. Using Selenium for JavaScript-rendered content.")
            content = get_content_with_selenium(url, headers)
            print("check is content_valid")
            return content

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        # Optionally try with Selenium
        print("Trying with Selenium due to HTTP error.")
        content = get_content_with_selenium(url, headers)
        return content
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        # Optionally try with Selenium
        print("Trying with Selenium due to Request error.")
        content = get_content_with_selenium(url, headers)
        return content
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

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
    # For example, check if there's meaningful content in the <body>
    body_text = soup.body.get_text(strip=True) if soup.body else ''
    if len(body_text) > 200:  # Threshold for meaningful content
        return True
    else:
        return False

def get_content_with_selenium(url, headers):
    """
    Fetches the content of a webpage using Selenium to render JavaScript.

    Parameters:
    - url (str): The URL of the webpage to fetch.
    - headers (dict): A dictionary of HTTP headers.

    Returns:
    - content (str): The HTML content of the webpage.
    """
    # Implement random delay to mimic human behavior
    time.sleep(random.uniform(1, 5))

    options = Options()
    options.add_argument('--headless')
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
        # Wait for the page to fully render
        time.sleep(random.uniform(2, 5))

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


def get_first_n_words(text, n=50):
    words = text.split()[:n]
    return ' '.join(words) + '...'


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return words


def get_high_frequency_words(text, num=10):
    words = clean_text(text)
    words = [word for word in words if is_valid_keyword(word)]
    counter = Counter(words)
    most_common = counter.most_common(num)
    return most_common


def get_high_frequency_phrases(text, num=10):
    words = clean_text(text)
    # Construct candidate phrases from adjacent cleaned words
    phrases = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    # Filter out any non-valid phrases
    valid_phrases = [phrase for phrase in phrases if is_valid_keyword(phrase)]
    counter = Counter(valid_phrases)
    most_common = counter.most_common(num)
    return most_common


def get_lsi_keywords(text, num=10):
    if isinstance(text, str):
        documents = [text]


    documents = re.split(r'(?<=[.!?])\s*', text)
    if not documents:
        raise ValueError("The documents are empty after preprocessing.")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    if len(vectorizer.vocabulary_) == 0:
        raise ValueError("Vocabulary is empty.")
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    average_similarities = cosine_similarities.mean(axis=0)
    indices = average_similarities.argsort()[-num:]
    feature_names = vectorizer.get_feature_names_out()

    keywords = [feature_names[index] for index in indices[::-1] if is_valid_keyword(feature_names[index])]
    return keywords[:num]


def best_practices_for_blog(article_length):
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


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text
# Assuming your existing functions are defined here as they are

#Extraction of text when there is no content within a paragraph
def extract_text_from_html_no_p(html_content):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, 'html.parser')

    # Exclude unwanted elements
    for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
        script_or_style.decompose()

    # Extract text
    text = soup.get_text(separator=' ', strip=True)

    # Clean up the text
    text = ' '.join(text.split())

    return text


def streamlit_ui():     
    st.title("SEO Content Writer ")
    all_results = ""
   #We are just examining one URL for now 
   # num_urls = st.number_input("Which website do you want to analyze?", min_value=1, max_value=1, value=1)
    article_length = 0
   # if num_urls > 1:
    #    article_length = st.number_input("Enter the expected length of your article in words (minimum 1000 words):", min_value=1000, value=1000)
    num_urls = 1
    urls = [st.text_input(f"Please enter webpage URL for keyword analysis {i+1}:", key=i) for i in range(num_urls)]
    article_topic = st.text_input("Enter the topic for your article:")

    if st.button("Generate"):
        for i, url in enumerate(urls):
            if url:  # Ensure URL is not empty
                html_content = get_webpage_content(url)
                if html_content is None:
                    st.write("Failed to retrieve content, please check the URL and try again.")
                    continue  # Skip further processing for this URL
                text = extract_text_from_html(html_content)
                if len(text)<10:
                    text = extract_text_from_html_no_p(html_content)
                lsi_keywords = get_lsi_keywords(text)
                high_freq_keywords = get_high_frequency_words(text)
                high_freq_phrases = get_high_frequency_phrases(text)

                # Assuming high_freq_keywords and high_freq_phrases are lists of tuples like [('word', count), ...]
                keywords_string = ", ".join([f"{kw[0]} ({kw[1]})" for kw in high_freq_keywords])
                phrases_string = ", ".join([f"{phrase[0]} ({phrase[1]})" for phrase in high_freq_phrases])

                all_results = f"{phrases_string}"

                print(all_results)
                st.write(f"Analysis for URL: {url}")
                st.write(f"Text Preview: {get_first_n_words(text)}")
                st.write("LSI Keywords:", ", ".join(lsi_keywords))
                st.write("High Frequency Keywords:", ", ".join([f"{kw[0]} ({kw[1]})" for kw in high_freq_keywords]))
                st.write("High Frequency Two-Word Phrases:", ", ".join([f"{phrase[0]} ({phrase[1]})" for phrase in high_freq_phrases]))
                st.write("=" * 50)

        if num_urls >= 2:
            # Aggregate analysis logic here
            # You would need to adapt this part to display aggregated results in the Streamlit app
            pass

        # Display best practices for blog articles
        if article_length:
            best_practices = best_practices_for_blog(article_length)
            st.write(f"Best Practices for {article_length}-word Blog Articles:")
            for key, value in best_practices.items():
                if key != 'tips':
                    st.write(f"{key.replace('_', ' ').capitalize()}: {value}")
                else:
                    st.write("Tips:")
                    for tip in value:
                        st.write(f"- {tip}")
    

        # Generate summary with OpenAI GPT
        blog_content = summary(article_topic, all_results)

        # Create a dynamic filename based on the URL
        dynamic_filename = create_filename_from_url(url)

        # Save the blog content with a dynamic filename
        save_data_to_json(blog_content, filename=dynamic_filename)

        st.write(blog_content)

if __name__ == "__main__":

    if authenticate_user():  # If authenticated successfully, run the app  
        streamlit_ui()
