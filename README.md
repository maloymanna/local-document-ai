# Local Document AI Assistant

An offline, document-aware AI assistant that runs locally using open-source LLMs like Qwen or DeepSeek (in GGUF format). Built with LangChain, FAISS, and Streamlit for local RAG (Retrieval-Augmented Generation).

---

## 🚀 Features

- Upload up to 5 documents at once (PDF, DOCX, TXT)
- Ask questions about your documents
- Uses a **locally run LLM** (no internet/cloud needed)
- Runs on **Windows, Mac, Linux**
- Supports **Qwen/DeepSeek-like GGUF models**

---

## 🧰 Requirements

- Python 3.9–3.11
- No admin rights required
- Windows 10+ recommended
- ~16GB RAM suggested for larger models

---

## 🔽 Model Setup

You must provide a **GGUF model file** (e.g., Q4_K_M quantized) and place it in the `models/` folder.

Example models:
- [Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF )
- [Qwen2-7B-Instruct-GGUF](https://huggingface.co/qwen/Qwen2-7B-Instruct-GGUF )

---

## 🛠️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/maloymanna/local-document-ai.git 
   cd local-document-ai
   ```

2. Create virtual environment:
   ```bash
   python -m venv env
   source env/Scripts/activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your `.gguf` model in `models/` folder. 

5. Run the app:
   ```bash
   streamlit run app.py
   ```
   Open in browser: http://localhost:8501 

