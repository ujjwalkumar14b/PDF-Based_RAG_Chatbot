import os
import secrets
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
chat_histories: dict[str, InMemoryChatMessageHistory] = {}
conversations: dict[str, RunnableWithMessageHistory] = {}

def get_pdfs_text(pdf_docs):
    text = []

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text.append(page.extract_text() or "")

    return "\n".join(text)

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()

    return chat_histories[session_id]

def get_conversation_chain(vector_store):
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.3,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant that answers questions
based on the provided document context.

Use the context below to answer the user's question.

If the answer cannot be found in the provided context,
clearly say that you don't know based on the provided documents.

Context:

{context}
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_function(inputs):
        docs = retriever.invoke(inputs["input"])
        messages = prompt.invoke(
            {
                "context": format_docs(docs),
                "chat_history": inputs.get("chat_history", []),
                "input": inputs["input"],
            }
        )
        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "context": docs,
        }

    rag_chain = RunnableLambda(rag_function)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def get_browser_session_id():
    if "session_id" not in session:
        session["session_id"] = secrets.token_urlsafe(32)

    return session["session_id"]

@app.get("/")
def index():
    get_browser_session_id()
    return render_template("index.html")

@app.post("/upload")
def upload_documents():
    pdf_docs = [pdf for pdf in request.files.getlist("pdfs") if pdf.filename]

    if not pdf_docs:
        return jsonify(error="Choose at least one PDF file to upload."), 400

    try:
        raw_text = get_pdfs_text(pdf_docs)
    except PdfReadError:
        app.logger.exception("A PDF could not be read")
        return jsonify(error="One or more PDFs could not be read. Check the files and try again."), 400

    text_chunks = get_text_chunks(raw_text)
    if not text_chunks:
        return jsonify(error="No readable text was found in the uploaded PDFs."), 400

    vector_store = get_vectorstore(text_chunks)
    session_id = get_browser_session_id()
    conversations[session_id] = get_conversation_chain(vector_store)
    chat_histories[session_id] = InMemoryChatMessageHistory()

    return jsonify(
        message=f"Processed {len(pdf_docs)} PDF file(s). You can now ask questions.",
        document_count=len(pdf_docs),
    )

@app.post("/chat")
def chat():
    payload = request.get_json(silent=True)
    question = payload.get("question", "").strip() if isinstance(payload, dict) else ""
    if not question:
        return jsonify(error="Enter a question about your PDFs."), 400

    session_id = get_browser_session_id()
    conversation = conversations.get(session_id)
    if conversation is None:
        return jsonify(error="Upload and process PDF documents before asking a question."), 400

    response = conversation.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    return jsonify(answer=response["answer"])

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
