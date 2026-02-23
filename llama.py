import os
import time
import warnings
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama



from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings



warnings.filterwarnings("ignore")
load_dotenv()

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

app = Flask(__name__)
CORS(app)

# --- Prompt template for RAG ---
custom_prompt_template = """You are a helpful assistant for question answering over a set of documents.

First, check the provided context.

- If the context contains relevant information, use ONLY that context to answer.
- If the context is insufficient or empty, still attempt to generate an answer from your own knowledge, but clearly state at the beginning:
  "Context is not available in data. Based on general knowledge, here is the answer:"

Context:
{context}

Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


# custom_prompt_template = """You are a helpful assistant for question answering over a set of documents.

# First, check the provided context.

# - If the context contains relevant information, use ONLY that context to answer.
# - If the context is insufficient or empty, still attempt to generate an answer from your own knowledge, but clearly state at the beginning:
#   "Context is not available in data. Based on general knowledge, here is the answer:"

# Context:
# {context}

# Question: {question}

# Instructions:
# - Give the answer in exactly 5 points.
# - Use bullet points or numbering.
# - Keep each point clear and short.

# Helpful answer:
# """



def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )

def load_llm():
    """
    Local model from Ollama.
    Make sure ollama is running and llama3.2 is pulled.
    """
    return ChatOllama(
        model="llama3.2",
        temperature=0.1,
        num_predict=500,
    )


# def load_llm():
#     """
#     OpenAI chat model for answers.
#     You can swap to 'gpt-4o' for higher quality or 'gpt-4o-mini' for cost/speed.
#     """
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.1,
#         max_tokens=500,
#     )
#     return llm

# ---------------------------
# Embeddings (local, HF)
# ---------------------------
# We will use a sentence-transformers model locally.
# "all-MiniLM-L6-v2" is ~384 dimensions, small & fast.
# IMPORTANT: we must use this SAME embedder for BOTH:
# - creating the FAISS index
# - loading/querying FAISS later



def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# def get_embedder():
#     return OpenAIEmbeddings(model="text-embedding-3-large")


def create_vector_db():
    """
    Build FAISS index from PDFs in DATA_PATH using HuggingFaceEmbeddings.
    Call this when PDFs change OR when FAISS load fails.
    """
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embedder()


    db = FAISS.from_documents(chunks, embeddings)
    # persist to disk
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)

def build_rag_chain(llm, prompt, db):
    """
    Build retrieval-augmented pipeline:
    question -> retrieve docs -> inject into prompt -> llm -> text
    """
    retriever = db.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def qa_bot():
    """
    Load FAISS index. If it fails (first run, dim mismatch, etc.),
    rebuild it with the current HuggingFaceEmbeddings model.
    """
    embeddings = get_embedder()

    try:
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"[FAISS] Load failed: {e}\nRebuilding index at {DB_FAISS_PATH} ...")
        create_vector_db()
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    llm = load_llm()
    prompt = set_custom_prompt()
    return build_rag_chain(llm, prompt, db)

def final_result(query: str) -> str:
    """
    Ask one question to the QA bot and return plain answer text.
    """
    qa = qa_bot()
    answer_text = qa.invoke(query)
    return answer_text

# ---------------- Flask app ----------------


@app.route("/")
def index():
    # This assumes you serve the HTML from a template file open_ai_trail.html.
    # Branding in that file should already say "sklassics".
    return render_template("open_ai_trail.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    query = data["query"]

    start = time.time()
    answer_text = final_result(query)
    end = time.time()

    response_time = end - start

    # same formatting strategy you had before
    response_time_line = f"response time - {response_time} seconds \n"
    raw_response = response_time_line + answer_text
    steps = raw_response.split("\n")
    formatted_response = "<br>".join(steps)

    return jsonify({"result": formatted_response})

@app.route("/reset", methods=["POST"])
def reset():
    """
    Frontend calls this when user clicks 'Clear Chat'.
    We don't actually persist per-user chat server-side yet,
    so we just return success so UI behaves.
    """
    return jsonify({"status": "ok", "message": "history cleared on server (placeholder)"}), 200

if __name__ == "__main__":
    # If FAISS folder doesn't exist yet, build it first so queries won't crash
    if not os.path.exists(DB_FAISS_PATH):
        create_vector_db()

    app.run(debug=True,host="0.0.0.0",port=8089)
