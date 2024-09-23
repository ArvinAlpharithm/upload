import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load environment variables for Groq API key
load_dotenv(override=True)

# Initialize the Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768",  # You can replace with your desired model
    api_key=os.getenv("GROQ_API_KEY")  # Load the API key from environment
)

# Streamlit app setup
st.set_page_config(layout='wide')

# CSV file uploader
input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    try:
        # Read the uploaded CSV file into a dataframe
        data = pd.read_csv(input_csv)
        st.info("CSV Uploaded Successfully")
        st.dataframe(data, use_container_width=True)

        # Create the CSV agent using the Groq LLM
        csv_agent = create_csv_agent(
            llm,
            input_csv,  # Pass the CSV file object directly
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=False  # Set to True if you want advanced code execution
        )
        
        # Text area for the user to enter their query
        input_text = st.text_area("Enter your query")

        if input_text:
            if st.button("Chat with CSV"):
                with st.spinner("Generating answer..."):
                    # Get the response from the CSV agent based on the query
                    response = csv_agent(input_text)
                    st.write("### Answer:")
                    st.text(response)

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please upload a CSV file to start interacting with it.")
