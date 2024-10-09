import os
from dotenv import load_dotenv
from typing import Annotated 
from typing_extensions import TypedDict

# Load environment variables from .env file
load_dotenv()

# Importing necessary libraries from LangChain and LangGraph
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition    
from langchain_openai import AzureChatOpenAI

# Initialize Wikipedia and Arxiv tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

tools = [wiki_tool]

# Define state type for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the state graph for LangGraph
graph_builder = StateGraph(State)

# Initialize Azure OpenAI LLM model
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add nodes and edges to the graph builder for flow execution
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph for execution
graph = graph_builder.compile()

# Main execution flow for user input queries
if __name__ == "__main__":
    user_input = input("Enter your question: ")
    
    events = graph.stream(
        {"messages": [("user", user_input)]}, stream_mode="values"
    )

    for event in events:
        event["messages"][-1].pretty_print()