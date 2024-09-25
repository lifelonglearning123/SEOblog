import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Function to apply custom CSS to expand the width of the main content and the table
def set_custom_css():
    st.markdown(
        """
        <style>
        /* Increase the width of the main content */
        .main {
            max-width: 1200px;
            padding-left: 20px;
            padding-right: 20px;
        }
        /* Adjust the width of the data editor table */
        div[data-testid="stDataEditorContainer"] {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the custom CSS
set_custom_css()

# Generate a sample DataFrame with more rows
def generate_sample_data(start_date, start_time, days):
    start_datetime = datetime.combine(start_date, start_time)
    post_dates = [(start_datetime + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(days)]
    generated_content = [f"Generated content for post {i+1}" for i in range(days)]
    
    generated_df = pd.DataFrame({
        "postAtSpecificTime (YYYY-MM-DD HH:mm:ss)": post_dates,
        "content": generated_content,
        "link (OGmetaUrl)": "",  # Empty columns for links
        "imageUrls": "",
        "gifUrl": "",
        "videoUrls": ""
    })
    return generated_df

# Function to download the DataFrame as a CSV file
def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Create the Streamlit app
st.title("Editable Social Media Calendar")

# User input for start date
start_date = st.date_input("Select the start date for the calendar")

# User input for time (hours and minutes)
start_time = st.time_input("Select the start time for each post")

# Add slider for controlling the number of days
days = st.slider("Select number of days for the calendar", min_value=1, max_value=30, value=30)

# Generate the DataFrame
generated_df = generate_sample_data(start_date, start_time, days)

# Display the editable table
st.write("Below is the generated content. You can edit it as needed:")
edited_df = st.data_editor(generated_df, num_rows="dynamic", height=400)

# Provide a download button for the edited DataFrame
st.download_button(
    label="Download Social Media Calendar",
    data=download_csv(edited_df),
    file_name="social_media_content.csv",
    mime="text/csv"
)