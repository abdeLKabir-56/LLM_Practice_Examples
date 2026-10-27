from dotenv import load_dotenv
from pinecone import Pinecone , ServerlessSpec
import os,time
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


loader = UnstructuredLoader(
    file_path="./1_AI_Agents_and_Agentic_Workflows.txt",
    chunking_strategy = "basic",
    encoding="utf-8",
    max_characters = 1000000000
)

document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size= 1000,chunk_overlap=0)
texts = text_splitter.split_documents(documents=document)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5"
)
index = pc.Index(os.environ["INDEX_NAME"])
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

vectorstore.add_documents(texts)

if __name__ == "__main__":
    print("Ingesting ...")
    print(f"created {len(texts)} chunks")

