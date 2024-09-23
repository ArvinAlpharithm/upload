import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Hardcode the API Key for Groq
os.environ["GROQ_API_KEY"] = "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"

# Initialize the Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768"  # Replace with your desired model
)

# Streamlit app title
st.title("CSV Query Agent with Groq")

# Define the path to your CSV file
csv_file_path = "data.csv"  # Update the path as necessary

# Load the CSV content as a DataFrame
try:
    df = pd.read_csv(csv_file_path)
    st.write("Data Preview:")
    st.dataframe(df.head())  # Display the first few rows of the dataframe

    # Create the CSV agent using the file path instead of the DataFrame
    csv_agent = create_csv_agent(
        llm,
        csv_file_path,  # Pass the path to the CSV file
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True  # Enable for advanced code execution
    )

    # Input for user query
    user_query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if user_query:
            # Example query to interact with the data
            response = csv_agent(user_query)
            st.write("Response:")
            st.write(response)
        else:
            st.warning("Please enter a query to submit.")
except FileNotFoundError:
    st.error("Error: The specified CSV file could not be found.")
except pd.errors.EmptyDataError:
    st.error("Error: The CSV file is empty or cannot be parsed.")
except Exception as e:
    st.error(f"An error occurred: {e}")
