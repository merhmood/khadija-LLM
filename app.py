import os
from pathlib import Path
import fitz  # PyMuPDF
import faiss
import numpy as np
import gradio as gr

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# =========================
# Config
# =========================
PDF_DIR = "./company_docs"
INDEX_PATH = "./data/faiss.index"
TOP_K = 4

SYSTEM_PROMPT = """
You are a helpful company assistant.
Answer ONLY using the provided company documents.
If the answer is not found in the documents, say you don’t know.
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
    citations = set()

    for i in indices[0]:
        context.append(chunks[i])
        m = metadatas[i]
        citations.add(f"{m['file']} (page {m['page']})")

    return "\n\n".join(context), "\n".join(sorted(citations))

# =========================
# TGI Client
# =========================
client = InferenceClient(
    model="http://tgi:80"  # Docker service name
)

# =========================
# Streaming Chat
# =========================
def stream_chat(user_input):
    context, citations = retrieve_context(user_input)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Company documents:\n{context}"},
        {"role": "user", "content": user_input},
    ]

    stream = client.chat_completion(
        messages=messages,
        max_tokens=250,
        temperature=0.3,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.get("content", "")
        partial += delta
        yield partial

    yield f"{partial}\n\n📚 Sources:\n{citations}"

# =========================
# Gradio UI (Streaming)
# =========================
with gr.Blocks(title="Company PDF Assistant") as demo:
    gr.Markdown("## 📄 Company Knowledge Assistant")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(
        placeholder="Ask a question about company documents…",
        scale=7
    )
    clear = gr.Button("Clear Chat")

    def respond(message, history):
        history.append((message, ""))
        for partial in stream_chat(message):
            history[-1] = (message, partial)
            yield history

    msg.submit(respond, [msg, chatbot], chatbot)
    clear.click(lambda: [], None, chatbot)

demo.launch(server_name="0.0.0.0", server_port=7860)
