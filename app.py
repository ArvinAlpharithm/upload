import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq

# Load environment variables from .env (like GROQ_API_KEY)
load_dotenv(override=True)

# Set up the Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")  # Store your Groq API key in a .env file
llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key=groq_api_key)

# Function to chat with the CSV file using PandasAI
def chat_with_csv(df, prompt):
    # Create a SmartDataframe and pass the Groq LLM
    smart_df = SmartDataframe(df, config={"llm": llm})
    
    # Use the PandasAI SmartDataframe to process the prompt
    result = smart_df.chat(prompt)
    
    return result

# Streamlit app setup
st.set_page_config(layout='wide')

# File uploader to upload CSV files
input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

    with col2:
        st.info("Chat Below")
        input_text = st.text_area("Enter your query")

        if input_text is not None:
            if st.button("Chat with CSV"):
                with st.spinner("Generating answer..."):
                    result = chat_with_csv(data, input_text)
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                    else:
                        # If result is not a DataFrame, display it as text
                        st.text(result)
