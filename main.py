from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# define the model
model = OllamaLLM(model="llama3.2")

# create a template (specify what we want from the model to do)

template =""" 
you are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# create a prompt template

prompt = ChatPromptTemplate.from_template(template=template)

#create a chain (this is a combination of the prompt and the model)

chain = prompt | model

while True:
    question = input("Ask a question about the pizza restaurant (q to quit): ")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)

