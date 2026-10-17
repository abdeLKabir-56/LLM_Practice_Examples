from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from typing import Literal,Annotated
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


llm = init_chat_model(
    model="llama3.2",
    model_provider="ollama",
    )

class Message(BaseModel):
    message_type: Literal["emotional","logical"] = Field(..., description="Classify if the message requires an emotional or logical response")

#set up the state graph
class State(TypedDict):
    messages : Annotated[list, add_messages]
    message_type : str | None
    

def message_classifier(state: State):
    last_message = state["messages"][-1]
    llm_classifier = llm.with_structured_output(Message)
    result = llm_classifier.invoke([{
        "role": "system",
        "content": """classify the user message as either :
        - 'logical' : if the message ask for facts , information , logical analysis or reasoning
        - 'emotional' : if the message ask for feelings , emotions , empathy or support
        """},
        {
        "role": "user",
        "content": last_message.content
    }])
    return {"message_type": result.message_type}

def router(state: State):
    message_type = state.get("message_type","logical")
    if message_type == "emotional":
        return {"next": "emotional"}
    return {"next": "logical"}

def emotional_agent(state: State):
    last_message = state["messages"][-1]
    result = llm.invoke([{
        "role": "system",
        "content": """You are an empathetic and supportive assistant. 
        Your task is to provide emotional support, understanding, and encouragement to the user.
        """
    },
    {
        "role": "user",
        "content": last_message.content
    }])
    return {"messages" : [result]}

def logical_agent(state: State):
    last_message = state["messages"][-1]
    result = llm.invoke([{
        "role": "system",
        "content": """You are a logical and analytical assistant. 
        Your task is to provide factual information, logical reasoning, and analytical insights to the user.
        """
    },
    {
        "role": "user",
        "content": last_message.content
    }])
    return {"messages" : [result]}

# create a node = function
#def chatbot(state: State):
#    return {"messages" : [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

# add the node to the graph
graph_builder.add_node("classifier", message_classifier)
graph_builder.add_node("router", router)
graph_builder.add_node("emotional", emotional_agent)
graph_builder.add_node("logical", logical_agent)



#add edges to the graph
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

# define conditional edges based on the message type
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "emotional": "emotional",
        "logical": "logical",
    }
)

graph_builder.add_edge("emotional", END)
graph_builder.add_edge("logical", END)

# compile the graph
graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}
    while True:
        user_input = input("Ask a question about the pizza restaurant (q to quit): ")
        if user_input.lower() == "q":
            break
        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        state = graph.invoke(state)
        if state.get("messages") and len(state["messages"]) > 0:
            last_response = state["messages"][-1]
            print(f"Assistant: {last_response.content}")
            
if __name__ == "__main__":
    run_chatbot()