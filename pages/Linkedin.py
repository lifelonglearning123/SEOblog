import streamlit as st
import openai
from dotenv import load_dotenv
import os
import json

def get_latest_json_file():
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    if json_files:
        # Sort by the modification time and pick the latest file
        latest_file = max(json_files, key=os.path.getmtime)
        return latest_file
    return None

def load_data_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Example usage
latest_filename = get_latest_json_file()
if latest_filename:
    data = load_data_from_json(latest_filename)
else:
    print("No JSON files found.")


#Setup environment
load_dotenv()  # Load environment variables from .env files
openai.api_key = os.getenv("OPENAI_API_KEY")


#Loading of JSON file - which contains the summary file
transcription = data
st.session_state.youtube_summary = transcription

st.title("Article Summary")

if "youtube_summary" in st.session_state and st.session_state.youtube_summary:
    st.text_area("YouTube Summary: ", st.session_state.youtube_summary, height =200)

# Function to generate the LinkedIn article
def generate_article(
    transcription, article_length, post_type, post_style, writing_style, user_comment
):
    messages = [
        {
            "role": "system",
            "content": "You are an expert content writer specialized in creating engaging LinkedIn posts.",
        },
        {
            "role": "user",
            "content": f"""
Based on the following transcription from a YouTube video, write an article for a LinkedIn post.

Transcription:
{transcription}

User Comment:
{user_comment}

Instructions:
- Length: {article_length} words.
- Post type: {post_type}.
- Post style: {post_style}.
- Writing style: {writing_style}.

Please ensure the article is engaging and suitable for LinkedIn.

Begin the article below:
""",
        },
    ]

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=article_length * 4,
        temperature=0.7,
        top_p=1,
        n=1,
        stop=None,
    )

    article = response.choices[0].message.content.strip()

    return article

# User Inputs
article_length = st.slider(
    "Select the article length (in words)",
    min_value=10,
    max_value=450,
    value=450,
    step=10,
)

post_type_options = ["Educational post", "Spicy take", "Head nod"]
post_type = st.selectbox(
    "What kind of LinkedIn post do you want to create?", post_type_options
)

post_style_options = [
    "The Personal Achievement Story",
    "The Industry Insight Revelation",
    "The Problem-Solving Showcase",
    "The Mentorship Magic Story",
    "The Hot Industry Take",
]
post_style = st.selectbox("What kind of LinkedIn post style do you want?", post_style_options)

writing_style_options = [
    "Direct and no-nonsense approach",
    "Conversational Insight and Value-Focused",
    "Concise Wisdom and Thought-Provoking",
    "Actionable Frameworks and Conversational Learning",
]
writing_style = st.selectbox("What writing style do you want?", writing_style_options)

# Add a text input for user comment
user_comment = st.text_input(
    "Enter a comment to guide the article (optional)", 
    placeholder="E.g., Focus on how this can help new entrepreneurs"
)
   
# Generate Article Button
if st.button("Generate Article"):
    if st.session_state.youtube_summary:
        with st.spinner("Processing..."):
            article = generate_article(
                    transcription,
                    article_length,
                    post_type,
                    post_style,
                    writing_style,
                    user_comment  # Include the user comment in the prompt
                )
            st.success("Article generated!")
            st.write(article)
            st.download_button(
                label="Download Article",
                data=article,
                file_name="article.txt",
                mime="text/plain",
            )
    else:
        st.error("Please enter a valid YouTube URL.")

