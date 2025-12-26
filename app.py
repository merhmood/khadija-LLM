import os
from pathlib import Path
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import gradio as gr

# =========================
# Config
# =========================
PDF_DIR = "./company_docs"
INDEX_PATH = "./data/faiss.index"
TOP_K = 4
HF_TOKEN = os.getenv("HF_TOKEN")  # Set as environment variable in production
MODEL_ID = "CohereLabs/command-a-reasoning-08-2025:cohere"

SYSTEM_PROMPT = """
You are a Yobe Trust Fund company assistant.
During greetings or salutations, introduce yourself as the Yobe Trust Fund company assistant.
Answer ONLY using the provided company documents.
If the answer is not found in the documents, say you don’t know in a polite manner.
If the user question is too vague, ask them to clarify.
Be concise and factual.
"""

# =========================
# PDF Loading + Chunking
# =========================
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def load_pdfs(pdf_dir):
    chunks = []
    metadata = []

    for pdf in Path(pdf_dir).glob("*.pdf"):
        doc = fitz.open(pdf)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            page_chunks = chunk_text(text)
            for chunk in page_chunks:
                chunks.append(chunk)
                metadata.append({
                    "file": pdf.name,
                    "page": page_num
                })
    return chunks, metadata

print("📄 Loading PDFs...")
chunks, metadatas = load_pdfs(PDF_DIR)

# =========================
# Embeddings + FAISS
# =========================
print("🧠 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(INDEX_PATH):
    print("🔁 Loading FAISS index from disk...")
    index = faiss.read_index(INDEX_PATH)
else:
    print("⚙️ Creating FAISS index...")
    embeddings = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

def retrieve_context(query, top_k=TOP_K):
    q_vec = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(q_vec, top_k)

    context = []
    for i in indices[0]:
        # Avoid duplicates
        if chunks[i] not in context:
            context.append(f"[Doc: {metadatas[i]['file']}, Page: {metadatas[i]['page']}]\n{chunks[i]}")
    return "\n\n".join(context)


# =========================
# HuggingFace OpenAI-compatible client
# =========================
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# =========================
# Chat function
# =========================
def chat_fn(message, history=None):
    # history is managed by Gradio, can be ignored internally if you want
    context = retrieve_context(message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Company documents:\n{context}"}
    ]

    # If you want to include previous messages
    if history:
        messages += history

    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages
    )

    answer = completion.choices[0].message.content

    # Gradio handles the history, so just return the answer string
    return answer

# =========================
# Launch Gradio ChatInterface
# =========================
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Yobe Trust Fund Company Assistant",
    description="Ask questions about our company.",
    save_history=True
)

demo.launch(server_name="0.0.0.0", server_port=7860)
