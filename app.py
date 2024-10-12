import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from typing import Annotated
import re

# Load environment variables
load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT_NAME or not AZURE_OPENAI_API_VERSION:
    raise ValueError("Azure OpenAI API credentials are missing in .env file.")

# System prompt for the LLM
SYSTEM_PROMPT = "You are an IP specialist, and you will answer the given prompt by using your knowledge."

# Initialize AzureChatOpenAI model
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION
)

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Tool to fetch IP details using IPStack API
@tool
def fetch_ip_details_tool(ip_address: str):
    """
    Fetch details of a given IP address using the IPStack API.
    """
    API_URL = f"https://api.ipstack.com/{ip_address}?access_key=b68789c2a59492afff58e8658831ade8"
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch IP details"}

# Set of tools available
tools = [fetch_ip_details_tool]

# Utility function to detect if the user's message contains an IP address
def detect_ip_in_message(message: str) -> str:
    ip_regex = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    match = re.search(ip_regex, message)
    if match:
        return match.group(0)  # Return the IP address if found
    return None

# Handle tool calls in conversation
def handle_tool_calls(result, messages):
    for tool_call in result.tool_calls:
        print("Use Tool:", tool_call)
        
        # Find the appropriate tool by its name (case-insensitive match)
        selected_tool = {tool.name.lower(): tool for tool in tools}[tool_call["name"].lower()]
        
        # Invoke the selected tool with the arguments provided in the tool call
        tool_output = selected_tool.invoke(tool_call["args"])
        print(tool_output)
        
        # Append the tool output to the conversation as a message
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

# Create an LLM with tools
def chatbot(user_input: str, chat_history: list):
    # System message setup
    sys_msg = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Detect if an IP address is present in the user input
    ip_address = detect_ip_in_message(user_input)
    
    if ip_address:
        # Use the tool to fetch IP details
        ip_details = fetch_ip_details_tool.invoke({"ip_address": ip_address})
        response_message = f"Here are the details for the IP {ip_address}: {ip_details}"
    else:
        # If no IP address is found, proceed with the usual conversation flow
        messages = [HumanMessage(content=f"{SYSTEM_PROMPT}\nUser: {user_input}")]
        llm_response = llm(messages=messages)
        
        if not llm_response.content:
            response_message = "I'm sorry, I couldn't provide an answer."
        else:
            response_message = llm_response.content
    
    return response_message

# Flask route for chatbot page
@app.route('/')
def index():
    return render_template('chatbot.html')

# Flask route to handle user input (AJAX call)
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['message']
    chat_history = session.get('chat_history', [])
    
    # Add user input to chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Get the chatbot response
    response_message = chatbot(user_input, chat_history)
    
    # Add bot response to chat history
    chat_history.append({"role": "bot", "content": response_message})
    session['chat_history'] = chat_history
    
    return jsonify({'response': response_message})

# Main entry point for running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
