import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore



load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
llm = ChatOllama(model="qwen3:1.7b",temperature=0)

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)
# retreive the top 3 relevent chuncks
retreiver = vector_store.as_retriever(search_kwargs = {"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context :
    {context}
    Question : {question}
    Provide a detailed answer :
    """
)

def format_docs(docs):
    """Format retreived docs into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


# this impl without using  lcel
def retreival_chain_without_lcel(query:str):
    """
    A simple retreival chain without LCEL
    Manually retrieves documents , formats them, and generates a response.
    
    Limitations :
    - Manual step-by-step execution
    - No build-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # get relevant docs
    docs = retreiver.invoke(query)
    
    # Format the docs to string
    context = format_docs(docs=docs)
    
    # Format the prompt with context and question
    messages = prompt_template.format_messages(context=context , question=query)
    
    # invoke llm with formated messages
    response = llm.invoke(messages)
    # return result
    return response.content
    
#impl using lcel 
def retreival_chain_with_lcel():
    """
    A retreival chain using LCEL
    Returns a chain that can be invoked with {"questions" :"..."}
    advantages of using lcel :
    - Declarative and composable : Easy to chain operations with pipe operator (|)
    - Build in streaming : chain.stream() works out of the box
    - Build in async : chain.ainvoke() and chain.astream() available
    - Batch processing : chain.batch() for multiple inputs
    - Type safety: Better integration with langchain's type system
    - less code : mORE concise and readable
    - Reusable : Chain can be saved , shared ,and composed with other chains
    - Better debugging: langchain provides better observability tools
    """
    retreival_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retreiver | format_docs
        ) |
        prompt_template |
        llm |
        StrOutputParser()
    )
    return retreival_chain
    
    
def main():
    query = "Why might LangGraph be preferable for complex workflows?" 
    #print(retreival_chain_without_lcel(query=query) )
    chain = retreival_chain_with_lcel()  
    result = chain.invoke({
        "question" : query
    })
    print(result)

    



if __name__ == "__main__":
    main()