# AI Bills RAGBot

A Retrieval-Augmented Generation (RAG) chatbot built with Chainlit, Langchain, Qdrant, and OpenAI.  
This app answers questions about AI-related bills in the Philippines by searching PDFs as a knowledge base.

---

## Features

- PDF ingestion and chunking with Langchain  
- Vector storage and similarity search with Qdrant  
- OpenAI-powered answer generation  
- Web chat interface powered by Chainlit  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Qdrant instance URL and API key  
- OpenAI API key  

### Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   
2. Create a .env file based on .env.example and add your keys:
   OPENAI_API_KEY=your-openai-key
   QDRANT_URL=https://your-qdrant-instance.com
   QDRANT_API_KEY=your-qdrant-api-key
   COLLECTION_NAME=chainlit_rag_collection

3. Install dependencies:
   pip install -r requirements.txt


4. Ingest your PDFs to Qdrant (optional if you are running ingest on deployment):
   python ingest.py

Run the Chainlit app locally:
   chainlit run app.py
