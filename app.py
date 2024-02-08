import streamlit as st
import re
from bs4 import BeautifulSoup
from collections import Counter
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import openai
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def summary(full_response, all_results):
    #print("**************")
    full_response = "write articles on the following topic: " + full_response + "Use following keywords within the articles " + all_results + "Write a blog in 1500 words. Use descriptive and engaging title. Include target keyword in the title. Use subheadings to break the content. Include a call to action."
    #print(full_response)
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview", 
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


def get_webpage_content(url):
    try:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error occurred: {req_err}")



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
    documents = re.split(r'(?<=[.!?])\s*', text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
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
def get_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        return response.content
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")  # Use Streamlit's error message display
    except requests.exceptions.RequestException as req_err:
        st.error(f"Error occurred: {req_err}")  # Use Streamlit's error message display
    return None
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
    
        test = summary(article_topic, all_results)
        st.write(test)
if __name__ == "__main__":
    streamlit_ui()
