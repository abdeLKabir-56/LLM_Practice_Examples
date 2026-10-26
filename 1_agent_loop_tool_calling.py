from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_IERATIONS = 10
MODEL = "qwen3:1.7b"

load_dotenv()

@tool
def get_product_price(product:str)->float:
    """look up the price of the product in the catalog"""
    print(f"executing get product price for : {product}")
    prices = {"laptop" : 1993.4, "iphone 15 pro max" : 799.0 , "keyboard": 180.6}
    return prices.get(product,0)


@tool
def apply_discount(price: float , discount_tier : str) -> float:
    """ applying discount based on price of the product given and returns the final price"""
    discount_packs = {"gold" : 23 , "silver" :12 , "bronze" : 5}
    return round(price - (price * (discount_packs.get(discount_tier,0)/100)),2)


# -- agent loop --


@traceable(name="Langchain agent loop")
def run_agent(question : str):
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}",temperature=0)
    ## give the model witch tools we have
    llm_with_tools = llm.bind_tools(tools)
    messages =[
        SystemMessage(
            content=(
                "You are a helpfull shopping assistant ."
                "You have access to a product catalog tool"
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price . \n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exect price "
                "returned by get_product_price- do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply_discount tool. \n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]
    
    for iteration in range(1, MAX_IERATIONS + 1):
        print(f"ieteration : {iteration}")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"final message : {ai_message.content}") 
            return ai_message.content
        
        # process tool calls
        tool_call = tool_calls[0]
        tool_call_name = tool_call.get("name")
        tool_call_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        print(f" [Tool Selected] : {tool_call_name} with args: {tool_call_args}")
        
        tool_to_use = tools_dict.get(tool_call_name)
        if not tool_to_use:
            raise ValueError(f"tool {tool_to_use} not found!")
        observation = tool_to_use.invoke(tool_call_args)
        print(f"[Tool result] : {observation}")
        
        #trace back
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation),tool_call_id = tool_call_id)
        )
    print("Error : Max iterations reached without find final answer")
    return None


if __name__ == "__main__":
    run_agent("what's laptop price after appling gold tier ?");