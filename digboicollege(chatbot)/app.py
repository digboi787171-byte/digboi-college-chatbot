import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Digboi College Chatbot", page_icon="🎓")

st.title("🎓 Digboi College Chatbot")
st.write("Ask anything about Digboi College")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "digboi_vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

device = 0 if torch.cuda.is_available() else -1

qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512,
    device=device
)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

question = st.text_input("Ask your question:")

if question:
    with st.spinner("Thinking..."):
        answer = qa.run(question)
    st.success(answer)
