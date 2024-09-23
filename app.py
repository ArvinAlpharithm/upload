import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Fetch the API Key for Groq from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    st.error("Error: GROQ_API_KEY is not set in the environment.")
else:
    # Initialize the Groq LLM with the fetched API Key
    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",  # Replace with your desired model
        api_key=GROQ_API_KEY
    )


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
