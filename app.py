import os
import tempfile
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
import streamlit as st

# Load model
@st.cache_resource
def load_llm(model_path):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.1,
        max_tokens=2048,
        top_p=0.95,
        n_gpu_layers=0,  # CPU-only
        n_batch=512,
        callback_manager=callback_manager,
        verbose=False,
    )
    return llm

# Load documents
def load_docs(files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        ext = Path(file.name).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(temp_filepath)
        elif ext == ".docx":
            loader = Docx2txtLoader(temp_filepath)
        elif ext == ".txt":
            loader = TextLoader(temp_filepath)
        else:
            continue
        docs.extend(loader.load())
    return docs

# Split text
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    return texts

# Embed and store
def create_vectorstore(texts, embedding_model="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.from_documents(texts, embeddings)
    return db

# Main app
def main():
    st.set_page_config(page_title="Local Document AI", page_icon="📚")
    st.title("📚 Local Document AI Assistant")

    model_path = "models/Llama-3-8B-Instruct-Gradient-1048k-Q4_K_M.gguf"
    llm = load_llm(model_path)

    uploaded_files = st.file_uploader("Upload up to 5 documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing documents..."):
            docs = load_docs(uploaded_files)
            texts = split_text(docs)
            db = create_vectorstore(texts)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

            user_query = st.text_input("Ask something about your documents:")
            if user_query:
                with st.spinner("Thinking..."):
                    response = chain.run(user_query)
                    st.markdown(f"**Answer:** {response}")

if __name__ == "__main__":
    main()