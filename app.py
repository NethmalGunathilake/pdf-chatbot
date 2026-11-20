import streamlit as st

from rag import ask_llm, chunk_text, embed_texts, load_pdf, retrieve

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

# Index the PDF only once per uploaded file (not on every click)
if st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading and indexing your PDF..."):
        text = load_pdf(uploaded)
        if not text.strip():
            st.error("No text found. This may be a scanned PDF (images only).")
            st.stop()
        chunks = chunk_text(text)
        st.session_state.chunks = chunks
        st.session_state.vectors = embed_texts(chunks)
        st.session_state.file_name = uploaded.name
        st.session_state.messages = []

st.success(f"Ready: {len(st.session_state.chunks)} chunks indexed.")

# Show the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat box
question = st.chat_input("Ask a question about your PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            top = retrieve(question, st.session_state.chunks, st.session_state.vectors)
            try:
                answer, model = ask_llm(question, [c for c, _ in top])
            except Exception:  # noqa: BLE001
                answer = "Sorry, the AI service is busy right now. Please try again in a minute."
        st.markdown(answer)
        with st.expander("Sources used"):
            for chunk, score in top:
                st.caption(f"Match score: {score:.2f}")
                st.write(chunk)

    st.session_state.messages.append({"role": "assistant", "content": answer})
