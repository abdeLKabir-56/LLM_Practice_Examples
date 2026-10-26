from dotenv import load_dotenv
from langsmith import traceable

# ollama python sdk
from ollama import chat
import re
import inspect

MAX_IERATIONS = 10
MODEL = "qwen3:1.7b"

load_dotenv()


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """look up the price of the product in the catalog"""
    print(f"executing get product price for : {product}")
    prices = {"laptop": 1993.4, "iphone 15 pro max": 799.0, "keyboard": 180.6}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """applying discount based on price of the product given and returns the final price"""
    discount_packs = {"gold": 23, "silver": 12, "bronze": 5}
    price = float(price)
    return round(price - (price * (discount_packs.get(discount_tier, 0) / 100)), 2)


# function calling - we will do it  manulay using prompt in this layer
# tools_for_llm = [
#    {
#             "type": "function",
#             "function": {
#                 "name": "get_product_price",
#                 "description" : "a tool to look up the price of the product in the catalog",
#                 "parameters" :{
#                     "type" : "object",
#                     "required" : ["product"],
#                     "properties" : {
#                         "product" :{ "type" : "string" , "description" : "it takes the product and give the price of this product"}
#                     }
#                 }
#             }
#         },
#      {
#             "type": "function",
#             "function": {
#                 "name": "apply_discount",
#                 "description" : "a tool to applying discount based on price of the product given and returns the final price",
#                 "parameters" :{
#                     "type" : "object",
#                     "required" : ["price" , "discount_tier"],
#                     "properties" : {
#                         "price" :{ "type" : "float" , "description" : "product price before applying the discount"},
#                         "discount_tier" :{ "type" : "string" , "description" : "product price discount based on this discount tier "}
#                     }
#                 }
#             }
#         },
# ]

tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}


def get_tool_description(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # orginal function before tracing
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_description(tools)
tool_names = " , ".join(tools.keys())


react_prompt = f"""
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought
"""


# langsmith trace
@traceable(name="Ollama chat traced", run_type="llm")
def ollama_chat_traced(model, messages, options=None):
    return chat(model=model, messages=messages, options=options)


# -- agent loop --


@traceable(name="Langchain agent loop")
def run_agent(question: str):

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_IERATIONS + 1):
        print(f"ieteration : {iteration}")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        
        final_answer_match = re.search(r"Final Answer: \s*(.+)",output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"final answer : {final_answer}")
            return final_answer
        
        action_match = re.search(r"Action: \s*(.+)",output)
        action_input_match = re.search(r"Action Input: \s*(.+)",output)
        
        if not action_match or not action_input_match :
            print(
                "ERROR : could not parse action/action input from llm outupt"
            )
            break
        


        # process tool calls
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        # tool_call_id = tool_call.get("id") in ollama chat tools we do not have ids

        print(f" [Tool Selected] : {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(" , ") ]
        args = [x.split("=",1)[-1].strip().strip("'\'") for x in raw_args]
        print(f"tools {tools} with {args}")
        if tool_name not in tools:
            observation = f"ERROR : tool {tool_name} not found . Available tools : {tools}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"[Tool result] : {observation}")

        # trace back
        scratchpad = f"{output}\nObservation : {observation}\nThought"
    print("Error : Max iterations reached without find final answer")
    return None


if __name__ == "__main__":
    run_agent("what's laptop price after appling gold tier ?")
