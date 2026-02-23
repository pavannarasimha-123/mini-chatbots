import os
from flask import Flask, render_template, request, jsonify

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

app = Flask(__name__)

DATA_PATH = "data"
DB_PATH = "vectorstore"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_db():
    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    embeddings = get_embeddings()

    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    db.save_local(DB_PATH)

    print("✅ Vector DB created")


# Always rebuild once (safe)
create_vector_db()

db = FAISS.load_local(
    DB_PATH,
    get_embeddings(),
    allow_dangerous_deserialization=True,
)

llm = Ollama(model="mistral")   # small + fast


@app.route("/")
def home():
    return render_template("open_ai_trail.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data["message"]

    docs = db.similarity_search(question, k=3)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Answer only from this context:

{context}

Question: {question}
"""

    answer = llm.invoke(prompt)

    return jsonify({"reply": answer})


if __name__ == "__main__":
    app.run(debug=True)