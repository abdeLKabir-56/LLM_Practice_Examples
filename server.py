from mcp.server import Server
from mcp.server.models import Tool , ToolRequest, ToolResponse
from pizza_details_vector import retriever

server = Server(name="pizza-restaurant-qa", description="A tool for answering questions about a pizza restaurant using reviews and a language model.")

@server.tool(name="restaurant-qa",description="a tool that brings us relevant details about a pizza restaurant based on reviews and a question")
def restaurant_pizza_details(request: ToolRequest) -> ToolResponse:
    retriever_response = retriever.invoke(request.input)
    return ToolResponse(output=retriever_response)
    
    