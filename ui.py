import streamlit as st
from utils.loader import load_pdf
from utils.splitter import split_docs
from utils.vectorstore import create_vectorstore
from utils.llm import load_llm
from utils.prompt import get_prompt

st.set_page_config(page_title="AI Tutor Chatbot", layout="wide")

st.title("🤖 AI Tutor Chatbot")
st.write("Ask questions from your uploaded document")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load pipeline
    with st.spinner("Processing document..."):
        docs = load_pdf("temp.pdf")
        chunks = split_docs(docs)
        db = create_vectorstore(chunks)
        llm = load_llm()
        prompt_template = get_prompt()

    st.success("Chatbot ready!")

    # Chat input
    query = st.text_input("Ask your question")

    if query:
        results = db.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in results])

        if len(context.strip()) < 100:
            st.error("This question is outside the provided material.")
        else:
            prompt = prompt_template.format(context=context, question=query)
            response = llm.invoke(prompt)
            st.write("🤖 Answer:")
            st.success(response)