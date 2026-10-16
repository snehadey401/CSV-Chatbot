from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma Database
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k":3})

# Load Ollama model
llm = ChatOllama(model="llama3")

print("CSV Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # Search relevant documents
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant.

Answer ONLY using the information below.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nBot:", response.content)
    print("-"*60)