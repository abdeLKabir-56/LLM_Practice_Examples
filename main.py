from dotenv import load_dotenv
from typing import List  
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch
import os
from pydantic import BaseModel , Field




load_dotenv()

tavily = TavilyClient()

class Source(BaseModel):
    """
    schema for a source used by the agent
    """
    url : str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """
    schema for agent response with answer and sources
    """
    answer : str = Field(description="The agent's Answer to the query")
    sources : List[Source] = Field(default_factory=list,description="list of sources used to generate the answer")

# create a basic exemple of tool
@tool
def search(query : str) -> str:
    """
    Tool that searches over the internet

    Args:
        query (str): the query to search for

    Returns:
        str: the search result
    """
    print(f"searching for {query}")
    return tavily.search(query=query)

@tool
def search_jobs(query : str ) -> str:
    """
    tool that searches over the internet for software engieer jobs professionals in morocco 

    Args:
        query (str): the query to search for

    Returns:
        str: the search result
    """
    print(f"searching for : {query}")
    return tavily.search(query=query)

# our llm   
llm = ChatOllama(
    model= "qwen3:4b",
    temperature=0
)

# list our tools
tools = [search,search_jobs,TavilySearch()]
# create our agent 
agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)

def run_agent():
    while True:
        question = input("ask ur question (q for exit): ")
        if question.lower() == "q":
            break
        result = agent.invoke({"messages" : HumanMessage(content=question)})
        print(result.get("structured_response"))

def main():
    print("hello world")
    run_agent()
    
if __name__ == "__main__":
    main()