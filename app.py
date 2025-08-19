import streamlit as st
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import docx2txt
import pypdf
import re
import io
import json

st.set_page_config(page_title="Document Chat", page_icon="📄", layout="wide")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = CHUNK_SIZE
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = CHUNK_OVERLAP
if "retrieve_count" not in st.session_state:
    st.session_state.retrieve_count = 6
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "processed" not in st.session_state:
    st.session_state.processed = False


def is_valid_api_key(api_key):
    return api_key and len(api_key) > 20


def load_pdf(file_content, filename):
    pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
    pages = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"text": text, "metadata": {"source": filename, "page": i + 1}})
    return pages


def load_docx(file_content, filename):
    text = docx2txt.process(io.BytesIO(file_content))
    return [{"text": text, "metadata": {"source": filename}}]


def load_txt(file_content, filename):
    text = file_content.decode('utf-8')
    return [{"text": text, "metadata": {"source": filename}}]


def process_documents(uploaded_files):
    all_chunks = []

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            try:
                if file_extension == '.pdf':
                    pages = load_pdf(file_content, uploaded_file.name)
                elif file_extension in ['.docx', '.doc']:
                    pages = load_docx(file_content, uploaded_file.name)
                elif file_extension == '.txt':
                    pages = load_txt(file_content, uploaded_file.name)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    continue

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len
                )

                for page in pages:
                    splits = text_splitter.create_documents([page["text"]])
                    for i, chunk in enumerate(splits):
                        chunk.metadata = page["metadata"]
                        chunk.metadata["chunk_id"] = i + 1
                        all_chunks.append(chunk)

                st.session_state.documents.append(uploaded_file.name)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    if all_chunks:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(all_chunks, embeddings)

        st.subheader("Sample Chunks")
        for i, chunk in enumerate(all_chunks[:3]):
            st.markdown(f"**Chunk {i+1}:** `{chunk.metadata}`")
            st.code(chunk.page_content[:500] + ("..." if len(chunk.page_content) > 500 else ""))

        condense_question_prompt = PromptTemplate.from_template("""
        Given the following conversation and a follow-up question, rephrase the follow-up question 
        to be a standalone question that captures all relevant context from the conversation.

        Chat History:
        {chat_history}

        Follow Up Question: {question}

        Standalone Question:
        """)

        qa_prompt = PromptTemplate.from_template("""
        You are a helpful and detailed document analysis assistant.

        When answering:
        - Provide inline citations like (DocumentName, Page X, Chunk Y)
        - Include specific document metadata
        - Clarify if multiple sources are used
        

        Context:
        {context}

        Question: {question}

        Answer:
        """)

        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=st.session_state.api_key,
                temperature=st.session_state.temperature
            )
        except Exception as e:
            st.error("Error initializing Gemini: " + str(e))
            return False

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_count}),
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )

        st.session_state.processed = True
        return True
    return False


def generate_response(user_question):
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation.invoke({"question": user_question})
            return response["answer"]
        except Exception as e:
            if "quota" in str(e).lower():
                return "API quota exceeded. Please check your Google Gemini API usage."
            elif "invalid" in str(e).lower():
                return "Invalid API key or unauthorized request."
            return f"Unexpected error: {str(e)}"


st.title("📄 Document Chat with RAG")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key

    if "api_key" in st.session_state and is_valid_api_key(st.session_state.api_key):
        st.success("API Key set successfully!")
    else:
        st.warning("Please enter a valid Google Gemini API Key")

    with st.sidebar.expander("Advanced Settings"):
        st.subheader("RAG Parameters")
        chunk_size = st.slider("Chunk Size", 500, 2000, CHUNK_SIZE, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, CHUNK_OVERLAP, step=50)
        retrieve_count = st.slider("Retrieved Chunks", 2, 10, 6, step=1)

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and "api_key" in st.session_state and is_valid_api_key(st.session_state.api_key):
        if st.button("Process Documents"):
            if process_documents(uploaded_files):
                st.success(f"Successfully processed {len(st.session_state.documents)} documents")
            else:
                st.error("Failed to process documents")

    if st.session_state.documents:
        st.subheader("Processed Documents")
        for doc in st.session_state.documents:
            st.write(f"- {doc}")

    if st.session_state.processed and st.button("Clear Conversation"):
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        if st.download_button("Download Chat History", data="\n\n".join(st.session_state.chat_history),
                              file_name="chat_history.txt", mime="text/plain"):
            st.success("Chat history downloaded")

if not st.session_state.processed:
    st.info("Please upload and process documents to start chatting")
else:
    st.subheader("Chat with your Documents")

    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user" if i % 2 == 0 else "assistant"):
            st.write(message)

    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        st.session_state.chat_history.append(user_question)

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            response = generate_response(user_question)
            st.write(response)
            st.session_state.chat_history.append(response)

st.markdown("---")
st.caption("Document Chat with RAG")
