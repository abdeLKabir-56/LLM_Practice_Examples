from dotenv import load_dotenv
from langsmith import traceable
# ollama python sdk
from ollama import chat

MAX_IERATIONS = 10
MODEL = "qwen3:1.7b"

load_dotenv()

@traceable(run_type="tool")
def get_product_price(product:str)->float:
    """look up the price of the product in the catalog"""
    print(f"executing get product price for : {product}")
    prices = {"laptop" : 1993.4, "iphone 15 pro max" : 799.0 , "keyboard": 180.6}
    return prices.get(product,0)


@traceable(run_type="tool")
def apply_discount(price: float , discount_tier : str) -> float:
    """ applying discount based on price of the product given and returns the final price"""
    discount_packs = {"gold" : 23 , "silver" :12 , "bronze" : 5}
    return round(price - (price * (discount_packs.get(discount_tier,0)/100)),2)


tools_for_llm = [
   {
            "type": "function",
            "function": {
                "name": "get_product_price",
                "description" : "a tool to look up the price of the product in the catalog",
                "parameters" :{
                    "type" : "object",
                    "required" : ["product"],
                    "properties" : {
                        "product" :{ "type" : "string" , "description" : "it takes the product and give the price of this product"}
                    }
                }
            }
        },
     {
            "type": "function",
            "function": {
                "name": "apply_discount",
                "description" : "a tool to applying discount based on price of the product given and returns the final price",
                "parameters" :{
                    "type" : "object",
                    "required" : ["price" , "discount_tier"],
                    "properties" : {
                        "price" :{ "type" : "float" , "description" : "product price before applying the discount"},
                        "discount_tier" :{ "type" : "string" , "description" : "product price discount based on this discount tier "}
                    }
                }
            }
        },
]


# langsmith trace
@traceable(name="Ollama chat traced" , run_type="llm")
def ollama_chat_traced(messages):
    return chat(model=MODEL,tools=tools_for_llm,messages=messages)

# -- agent loop --


@traceable(name="Langchain agent loop")
def run_agent(question : str):
    tools_dict = {
        "get_product_price" : get_product_price,
        "apply_discount" : apply_discount
    }

    messages =[
        {
            "role" : "system",
            "content" : """
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
            """
        },
        {
            "role" : "user",
             "content" : question
        }
    ]

    
    for iteration in range(1, MAX_IERATIONS + 1):
        print(f"ieteration : {iteration}")
        response = ollama_chat_traced(messages = messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"final message : {ai_message.content}") 
            return ai_message.content
        
        # process tool calls
        tool_call = tool_calls[0]
        tool_call_name = tool_call.function.name
        tool_call_args = tool_call.function.arguments
        #tool_call_id = tool_call.get("id") in ollama chat tools we do not have ids
        
        
        print(f" [Tool Selected] : {tool_call_name} with args: {tool_call_args}")
        
        tool_to_use = tools_dict.get(tool_call_name)
        if not tool_to_use:
            raise ValueError(f"tool {tool_to_use} not found!")
        observation = tool_to_use(**tool_call_args)
        print(f"[Tool result] : {observation}")
        
        #trace back
        messages.append(ai_message)
        messages.append(
            {
                "role" : "tool",
                "content" : str(observation)
            }
        )
    print("Error : Max iterations reached without find final answer")
    return None


if __name__ == "__main__":
    run_agent("what's laptop price after appling gold tier ?");