import os
from dotenv import load_dotenv
import chainlit as cl
from qdrant_client import QdrantClient
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Qdrant Client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Connect Qdrant
vectorstore = Qdrant(
    collection_name=COLLECTION_NAME,
    embeddings=OpenAIEmbeddings(),
    client=qdrant_client,
)

# Langchain Retrieval QA
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3, "score_threshold": 0.35})

# Prompt template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant. Answer the question ONLY using the information from the provided documents.
    If the answer is not contained in the documents, respond with "I’m sorry, I don’t know the answer to that question.".

    Documents:
    {context}

    Question:
    {question}

    Answer:
    """,
        input_variables=["context", "question"]
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# Chatbot starts chat session
@cl.on_chat_start
async def start():
    # Sends a starter message
    start_message = (
        "👋 Welcome to AI Bills RAGBot! "
        "This is your go-to assistant for questions about AI-related bills in the Philippines. "
        "How can I help you today?"
    )
    await cl.Message(content=start_message).send()

# Chainlit event
@cl.on_message
async def main(message: cl.Message):
    result = qa_chain.invoke({"query": message.content})

    answer = result["result"]
    sources = result["source_documents"]
    if not sources:  # threshold can be tuned
        answer = "❌ I’m sorry, I don’t know the answer to that question."

    await cl.Message(content=answer).send()

