import os
import re

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from google import genai
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Models to try, in order. If one is busy or retired, we move to the next.
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]


def load_pdf(file):
    """Read every page and join all the text into one string."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def chunk_text(text, max_chars=700, overlap=1):
    """Group whole sentences into chunks of up to ~700 characters."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], []
    for s in sentences:
        if current and len(" ".join(current + [s])) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap:]
        current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks


embedder = TextEmbedding("BAAI/bge-small-en-v1.5")


def embed_texts(texts):
    """Turn texts into unit-length vectors, shape (len(texts), 384)."""
    vectors = np.array(list(embedder.embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def retrieve(question, chunks, chunk_vectors, top_k=3):
    """Return the top_k (chunk, score) pairs closest in meaning to the question."""
    q_vec = embed_texts([question])[0]
    scores = chunk_vectors @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def ask_llm(question, context_chunks):
    """Send the question + retrieved chunks to Gemini and return (answer, model_used)."""
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}"""

    last_error = None
    for name in MODELS:
        try:
            response = client.models.generate_content(model=name, contents=prompt)
            return response.text, name
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise RuntimeError(f"All models failed. Last error: {last_error}")


if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    print(f"Ready: {len(chunks)} chunks. Type 'quit' to stop.")

    while True:
        question = input("\nAsk something about the PDF: ")
        if question.strip().lower() == "quit":
            break
        top = retrieve(question, chunks, vectors)
        answer, model = ask_llm(question, [c for c, _ in top])
        print(f"\nAnswer ({model}):\n{answer}")
