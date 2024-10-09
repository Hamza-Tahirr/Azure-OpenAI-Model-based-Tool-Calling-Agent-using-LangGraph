import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
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

# Setup Wikipedia tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
tools = [wiki_tool]

# Define a state structure for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LangGraph state graph
graph_builder = StateGraph(State)

# Define the system prompt for Azure OpenAI
SYSTEM_PROMPT = "You are an Intelligent History Chatbot, and you have to answer all the questions related to the History of the World."

# Initialize AzureChatOpenAI model
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION
)

# Create an LLM with tools
def chatbot(state: State):
    # Extract user's message correctly
    user_message = state["messages"][-1].content
    try:
        # Call Azure LLM with user's message
        messages = [HumanMessage(content=f"{SYSTEM_PROMPT}\nUser: {user_message}")]
        llm_response = llm(messages=messages)
        return {"messages": [HumanMessage(content=llm_response.content)]}
    except Exception as e:
        # Fallback to tools if LLM fails
        return {"messages": [wiki_tool.invoke(user_message)]}

# Add chatbot node to LangGraph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# Add Wikipedia tool node and integrate it with the chatbot
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Function to run the chatbot
def run_chatbot(user_input):
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]}, stream_mode="values"
    )
    for event in events:
        print(event["messages"][-1].content)

# Main entry point
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    run_chatbot(user_input)
