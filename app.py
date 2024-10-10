import os
import requests
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Ensure the API key is loaded
if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT_NAME or not AZURE_OPENAI_API_VERSION:
    raise ValueError("Azure OpenAI API credentials are missing in .env file.")

# Define a state structure for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    ip_address: str  # Ensure ip_address is included in the state

# Initialize LangGraph state graph
graph_builder = StateGraph(State)

# Define the system prompt for Azure OpenAI
SYSTEM_PROMPT = "You are an IP specialist and you will answer the given prompt by using your knowledge."

# Initialize AzureChatOpenAI model
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION
)

# Function to fetch IP details using IPStack API
def fetch_ip_details(ip_address: str):
    API_URL = f"https://api.ipstack.com/{ip_address}?access_key=b68789c2a59492afff58e8658831ade8"
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch IP details"}

# Create an LLM with tools
def chatbot(state: State):
    # Extract user's message correctly
    user_message = state["messages"][-1].content
    
    # Check if user wants IP details
    if "details of my IP" in user_message:
        ip_address = state["ip_address"]
        ip_details = fetch_ip_details(ip_address)
        return {"messages": [HumanMessage(content=str(ip_details))]}
    
    try:
        # Call Azure LLM with user's message
        messages = [HumanMessage(content=f"{SYSTEM_PROMPT}\nUser: {user_message}")]
        llm_response = llm(messages=messages)
        
        # Check if LLM response is satisfactory
        if not llm_response.content or "I'm sorry" in llm_response.content:
            # Fallback to IP details if LLM response is not satisfactory
            ip_address = state["ip_address"]
            ip_details = fetch_ip_details(ip_address)
            return {"messages": [HumanMessage(content=str(ip_details))]}
        
        return {"messages": [HumanMessage(content=llm_response.content)]}
    
    except Exception as e:
        # Log exception (optional)
        print(f"Error occurred: {e}")
        
        # Fallback to IP details in case of an exception
        ip_address = state["ip_address"]
        ip_details = fetch_ip_details(ip_address)
        return {"messages": [HumanMessage(content=str(ip_details))]}

# Add chatbot node to LangGraph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Function to run the chatbot
def run_chatbot():
    ip_address = input("Enter the IP address: ")
    user_input = input("Enter your query: ")
    
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)], "ip_address": ip_address}, stream_mode="values"
    )
    
    for event in events:
        print(event["messages"][-1].content)

# Main entry point
if __name__ == "__main__":
    run_chatbot()