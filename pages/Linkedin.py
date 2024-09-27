import streamlit as st
import openai
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env files (for OpenAI API key)
load_dotenv()  
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to get the latest JSON file in the current directory
def get_latest_json_file():
    """
    Returns the name of the most recently modified JSON file in the current directory.
    """
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    if json_files:
        # Sort files by modification time in descending order
        latest_file = max(json_files, key=os.path.getmtime)
        return latest_file
    return None

# Function to load data from a JSON file
def load_data_from_json(filename):
    """
    Loads data from a JSON file.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        The data loaded from the JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)

# Function to generate the LinkedIn article using OpenAI's API
def generate_article(transcription, article_length, post_type, post_style, writing_style, user_comment):
    """
    Generates an article based on the given parameters using OpenAI's language model.

    Args:
        transcription (str): The transcription text to base the article on.
        article_length (int): Desired length of the article in words.
        post_type (str): Type of LinkedIn post.
        post_style (str): Style of the post.
        writing_style (str): Writing style to use.
        user_comment (str): Additional comments to guide the article.

    Returns:
        str: The generated article text.
    """
    # Prepare the messages for the OpenAI ChatCompletion API
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

    # Call the OpenAI API to generate the article
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=article_length * 4,  # Estimate tokens needed (adjust as necessary)
        temperature=0.7,
        top_p=1,
        n=1,
        stop=None,
    )

    # Extract the generated article from the response
    article = response.choices[0].message.content.strip()

    return article

# Main function to run the Streamlit app
def main():
    """
    Main function to run the Streamlit app.
    """
    # Check if the latest JSON file exists
    latest_filename = get_latest_json_file()
    if latest_filename:
        # Load transcription data from the JSON file
        transcription = load_data_from_json(latest_filename)
        # Save the transcription in the session state
        st.session_state['youtube_summary'] = transcription
    else:
        transcription = None
        st.session_state['youtube_summary'] = None

    # Set the title of the app
    st.title("LinkedIn Article Generator")

    # Display the transcription in a text area if available
    if st.session_state.get('youtube_summary'):
        st.subheader("YouTube Summary")
        st.text_area("Transcription:", st.session_state['youtube_summary'], height=200)
    else:
        st.warning("No transcription data found. Please use the SEO tab to create an article.")
        return  # Stop the app here if no transcription is available

    # User Inputs
    st.subheader("Customise Your Article")

    # Article length slider
    article_length = st.slider(
        "Select the article length (in words)",
        min_value=50,
        max_value=450,
        value=450,
        step=50,
    )

    # Select post type
    post_type_options = ["Educational post", "Spicy take", "Head nod"]
    post_type = st.selectbox(
        "What kind of LinkedIn post do you want to create?",
        post_type_options,
        index=0,
    )

    # Select post style
    post_style_options = [
        "The Personal Achievement Story",
        "The Industry Insight Revelation",
        "The Problem-Solving Showcase",
        "The Mentorship Magic Story",
        "The Hot Industry Take",
    ]
    post_style = st.selectbox(
        "What kind of LinkedIn post style do you want?",
        post_style_options,
        index=0,
    )

    # Select writing style
    writing_style_options = [
        "Direct and no-nonsense approach",
        "Conversational insight and value-focused",
        "Concise wisdom and thought-provoking",
        "Actionable frameworks and conversational learning",
    ]
    writing_style = st.selectbox(
        "What writing style do you want?",
        writing_style_options,
        index=0,
    )

    # Text input for user comment
    user_comment = st.text_input(
        "Enter a comment to guide the article (optional)",
        placeholder="E.g., Focus on how this can help new entrepreneurs"
    )

    # Generate Article Button
    if st.button("Generate Article"):
        # Ensure transcription data is available
        if st.session_state.get('youtube_summary'):
            with st.spinner("Generating article..."):
                try:
                    # Call the function to generate the article
                    article = generate_article(
                        transcription=st.session_state['youtube_summary'],
                        article_length=article_length,
                        post_type=post_type,
                        post_style=post_style,
                        writing_style=writing_style,
                        user_comment=user_comment
                    )
                    print("Article generated successfully!")

                    # Display the generated article
                    st.subheader("Generated Article")
                    st.write(article)

                    # Provide a download button for the article
                    st.download_button(
                        label="Download Article",
                        data=article,
                        file_name="article.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"An error occurred while generating the article: {str(e)}")
        else:
            st.error("No transcription data available to generate the article.")

# Run the app
if __name__ == "__main__":
    main()
