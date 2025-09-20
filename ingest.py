import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Loads Keys
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "chainlit_rag_collection")

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Load Data Files (PDFs)
docs = []
for file in os.listdir("data"):
    path = os.path.join("data", file)
    if file.endswith(".txt") or file.endswith(".md"):
        docs.extend(TextLoader(path).load())
    elif file.endswith(".pdf"):
        docs.extend(PyPDFLoader(path).load_and_split())

# Split Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs_split = []

for doc in docs:
    docs_split.extend(splitter.split_documents([doc]))

# Embeddings
embeddings = OpenAIEmbeddings()
qdrant_store = Qdrant.from_documents(
    documents=docs_split,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
)

print(f"Upserted {len(docs_split)} chunks into Qdrant collection '{COLLECTION_NAME}'")