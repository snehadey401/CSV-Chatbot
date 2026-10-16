import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from csv_agent import process_calculation

# ---------------------------
# Streamlit Page
# ---------------------------
st.set_page_config(
    page_title="Student Performance Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Student Performance Chatbot")

# ---------------------------
# Load Embedding Model
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Load Chroma Database
# ---------------------------
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# Load Ollama Model
# ---------------------------
llm = ChatOllama(model="qwen2.5:3b")

# ---------------------------
# Chat History
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------
# User Input
# ---------------------------
question = st.chat_input("Ask anything about the dataset...")

if question:

    # Show User Message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    # Try Pandas Calculation First
    answer = process_calculation(question)

    # If not handled by Pandas, use RAG
    if answer is None:

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information below.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        answer = response.content

    # Show Bot Response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)