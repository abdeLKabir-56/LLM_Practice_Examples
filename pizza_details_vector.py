#impl vector search
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# our doc
df = pd.read_csv("pizza_sales_dataset.csv")
# embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db_pizza_details"

#not vectorize our documents if the db already exists
add_documents = not os.path.exists(db_location)


if add_documents:
    documents = []
    ids = []
    for i,row in df.iterrows():
        #pizza_id,order_id,pizza_name_id,quantity,order_date,order_time,unit_price,total_price,pizza_size,pizza_category,pizza_ingredients,pizza_name
        document = Document(
            page_content= str(row["quantity"]) + " " + str(row["unit_price"]) + " " + str(row["total_price"]) + " " + row["pizza_size"] + " " + row["pizza_category"] + " " + row["pizza_ingredients"] + " " + row["pizza_name"],
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

# create a vector store
vector_store = Chroma(
    collection_name="pizza_details",
    persist_directory=db_location,
    embedding_function=embeddings
    )

# we can add documents to the vector store if it doesn't already exist
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    

# make our embeddings used by the model to be used for searching
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10} # nbr of docs to lookup
)