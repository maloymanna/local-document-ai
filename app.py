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
from PIL import Image
import pytesseract
from langchain.docstore.document import Document

# --- CONFIG ---
MODEL_FOLDER = "models"
SUPPORTED_EXTENSIONS = [".gguf"]

# Get list of available models
def get_available_models():
    try:
        return sorted([
            f for f in os.listdir(MODEL_FOLDER)
            if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS)
        ])
    except FileNotFoundError:
        st.error("Models folder not found.")
        return []

# Load selected model
@st.cache_resource
def load_llm(model_path, max_tokens, temperature, top_p, n_batch, n_gpu_layers):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,
    )
    return llm

# OCR Function
def ocr_extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        return text.strip()
    except Exception as e:
        st.error(f"OCR failed: {str(e)}")
        return ""

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
    st.set_page_config(page_title="Local Document AI", page_icon="🧠", layout="centered")
    st.title("🧠 Local Document AI Assistant")

    # Sidebar - Model Selector
    st.sidebar.header("Model Settings")

    available_models = get_available_models()
    if not available_models:
        st.sidebar.warning("No GGUF models found in 'models/' folder.")
        st.stop()

    selected_model = st.sidebar.selectbox("Choose a model", available_models)
    model_path = os.path.join(MODEL_FOLDER, selected_model)

    st.sidebar.markdown("---")

    # Sidebar - Inference Settings
    st.sidebar.subheader("Inference Parameters")
    max_tokens = st.sidebar.slider(
        "Max Output Length",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum number of tokens (words/punctuation) to generate."
    )

    temperature = st.sidebar.slider(
        "Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Higher values produce more diverse outputs. Use 0.7+ for creative writing, 0.1–0.3 for factual responses."
    )

    top_p = st.sidebar.slider(
        "Top-p Sampling",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
        help="Controls diversity by limiting token selection to high-mass probability distribution."
    )

    n_batch = st.sidebar.slider(
        "Batch Size",
        min_value=32,
        max_value=2048,
        value=512,
        step=32,
        help="Number of tokens processed in one go. Higher values may improve speed but require more memory."
    )

    n_gpu_layers = st.sidebar.slider(
        "GPU Layers",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Number of model layers to run on GPU. Only works if CUDA is supported."
    )

    st.sidebar.markdown("---")

    # Load model
    try:
        llm = load_llm(model_path, max_tokens, temperature, top_p, n_batch, n_gpu_layers)
    except Exception as e:
        st.error("Error loading model. Make sure the file is valid and not corrupted.")
        st.exception(e)
        st.stop()

    # Upload documents
    uploaded_files = st.file_uploader("Upload up to 5 documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing documents..."):
            docs = load_docs(uploaded_files)
            texts = split_text(docs)
            db = create_vectorstore(texts)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    # Upload images
    uploaded_images = st.file_uploader("Upload PNG slides", type=["png"], accept_multiple_files=True)
    if uploaded_images:
        with st.spinner("Extracting text from slides..."):
            extracted_texts = []
            for image in uploaded_images:
                temp_dir = tempfile.TemporaryDirectory()
                temp_path = os.path.join(temp_dir.name, image.name)
                with open(temp_path, "wb") as f:
                    f.write(image.getvalue())
                text = ocr_extract_text_from_image(temp_path)
                if text:
                    extracted_texts.append(text)

            docs = [Document(page_content=text) for text in extracted_texts]
            texts = split_text(docs)
            db = create_vectorstore(texts)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    # Chat interface
    user_query = st.chat_input("Ask something about your documents:")
    if user_query:
        with st.spinner("Thinking..."):
            response = chain.run(user_query)
            st.markdown(f"**Answer:**\n\n{response}")

if __name__ == "__main__":
    main()