
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


# ---------------------------
# Load Environment Variables
# ---------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------
# Local Embeddings
# ---------------------------

class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

    def embed_documents(self, texts):
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text)
        return embedding.tolist()


# ---------------------------
# Streamlit Config
# ---------------------------

st.set_page_config(
    page_title="Book LLM",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book LLM with Chat Memory")


# ---------------------------
# Session State
# ---------------------------

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


# ---------------------------
# Upload PDF
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ---------------------------
# Process PDF
# ---------------------------

if uploaded_file and not st.session_state.pdf_loaded:

    with st.spinner("Reading PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    with st.spinner("Splitting into chunks..."):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

    with st.spinner("Creating embeddings..."):

        embeddings = SentenceTransformerEmbeddings()

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    with st.spinner("Loading LLM..."):

        llm = ChatOpenAI(
            model="nex-agi/nex-n2-pro:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
            temperature=0
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )

        st.session_state.conversation = conversation_chain
        st.session_state.pdf_loaded = True

    st.success("PDF processed successfully!")


# ---------------------------
# Display Chat History
# ---------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------
# Chat Input
# ---------------------------

if st.session_state.pdf_loaded:

    user_question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if user_question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = st.session_state.conversation.invoke(
                    {
                        "question": user_question
                    }
                )

                answer = result["answer"]

                st.markdown(answer)

                with st.expander("📄 Source References"):

                    sources = result["source_documents"]

                    for i, doc in enumerate(sources):

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )

                        st.markdown(
                            f"### Source {i+1} (Page {page+1 if isinstance(page, int) else page})"
                        )

                        st.write(
                            doc.page_content[:500] + "..."
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

else:
    st.info("Upload a PDF to begin chatting.")

