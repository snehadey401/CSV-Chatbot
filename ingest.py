import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Read CSV
df = pd.read_csv("data/StudentPerformanceFactors.csv")

documents = []

# Convert each row into a Document
for _, row in df.iterrows():
    text = ""

    for column in df.columns:
        text += f"{column}: {row[column]}\n"

    documents.append(Document(page_content=text))

print(f"Total Documents: {len(documents)}")

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma Database
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Database Created Successfully!")