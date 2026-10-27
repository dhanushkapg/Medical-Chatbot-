import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import create_embeddings
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

index_name = "medicalbot"

embedding_model = create_embeddings()

# Connect to the existing Pinecone index (populated by src/store_index.py)
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.6,
    api_key=os.getenv("OPENAI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        response = rag_chain.invoke({"input": question})
        answer = response["answer"]
        return jsonify({"answer": answer})

    except Exception:
        app.logger.exception("Error while generating the answer")
        return jsonify({
            "error": "Sorry, I couldn't generate an answer. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
