#impl vector search
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# our doc
df = pd.read_csv("realistic_restaurant_reviews.csv")
# embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

#not vectorize our documents if the db already exists
add_documents = not os.path.exists(db_location)


if add_documents:
    documents = []
    ids = []
    for i,row in df.iterrows():
        document = Document(
            page_content= row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]}, #additional data that we wana grab from the document when we do a search
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

# create a vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
    )

# we can add documents to the vector store if it doesn't already exist
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    

# make our embeddings used by the model to be used for searching
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # nbr of docs to lookup
)