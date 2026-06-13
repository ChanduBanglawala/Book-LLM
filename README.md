# 📚 Book LLM - Intelligent PDF Question Answering System

## Overview

Book LLM is an AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF books, research papers, academic notes, or documents and interact with them through natural language conversations.

The system extracts content from uploaded PDFs, converts the text into vector embeddings, stores them in a FAISS vector database, and retrieves relevant information to answer user questions accurately. It also maintains conversational memory, enabling context-aware interactions across multiple questions.

## Features

* Upload and analyze PDF documents
* Conversational Question Answering
* Retrieval-Augmented Generation (RAG)
* Semantic Search using Vector Embeddings
* Context-Aware Chat Memory
* Source Reference Tracking
* Multi-page Document Processing
* Fast Information Retrieval
* User-Friendly Streamlit Interface

## System Architecture

PDF Upload
↓
PDF Text Extraction (PyPDFLoader)
↓
Text Chunking (RecursiveCharacterTextSplitter)
↓
Embedding Generation (BAAI/bge-base-en-v1.5)
↓
FAISS Vector Database
↓
Retriever
↓
Large Language Model (OpenRouter LLM)
↓
Answer Generation with Sources

## Technologies Used

### Frontend

* Streamlit

### Large Language Model

* OpenRouter API
* Llama / DeepSeek / Other OpenRouter Models

### Embeddings

* BAAI/bge-base-en-v1.5

### Vector Database

* FAISS

### Frameworks

* LangChain

### Document Processing

* PyPDFLoader

### Memory

* ConversationBufferMemory

## How It Works

1. User uploads a PDF document.
2. The document is loaded and parsed.
3. Text is divided into manageable chunks.
4. Embeddings are generated for each chunk.
5. Chunks are stored in FAISS.
6. User asks a question.
7. Similar chunks are retrieved through semantic search.
8. Retrieved content is sent to the LLM.
9. The LLM generates a context-aware answer.
10. Source references are displayed to improve transparency.

## Project Objectives

* Build a practical Generative AI application.
* Implement Retrieval-Augmented Generation.
* Enable accurate document-based question answering.
* Reduce hallucinations through document grounding.
* Demonstrate the integration of LLMs with Vector Databases.

## Future Enhancements

* Multi-document support
* OCR for scanned PDFs
* Hybrid Search (Keyword + Semantic)
* Multi-LLM Support
* Voice-Based Interaction
* Citation Generation
* Cloud Deployment

## Learning Outcomes

This project demonstrates practical experience in:

* Generative AI
* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embedding Models
* Semantic Search
* LangChain
* Streamlit Application Development
* Conversational AI Systems
