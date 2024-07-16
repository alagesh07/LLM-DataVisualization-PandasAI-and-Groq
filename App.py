import pandas as pd
data = pd.read_csv("data.csv")
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_groq.chat_models import ChatGroq
import os

llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.environ["GROQ_API_KEY"]
)

from pandasai import SmartDataframe

df = SmartDataframe(data, config={"llm": llm})

def ai_interaction_loop():
    print("AI Interaction Loop. Type 'exit' to quit.")
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            try:
                # Interact with the SmartDataframe
                response = df.chat(user_input)
                print(f"{response}")
            except Exception as e:
                print(f"An error occurred: {e}")

# Start the AI interaction loop
ai_interaction_loop()
