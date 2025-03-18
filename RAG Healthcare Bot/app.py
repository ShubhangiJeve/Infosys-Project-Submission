import warnings
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def init_qa_system():
    # Set API Key (Replace with a valid Groq API Key)
    API_KEY = "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"
    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=API_KEY)

    # Load and process the PDF file (ensure health.pdf exists)
    pdf_path = "health.pdf"
    try:
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
    except Exception as e:
        raise ValueError(f"Error loading PDF file: {e}")

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(data)
    
    # If no documents were extracted, use a dummy fallback document
    if not documents:
        print("No documents loaded from PDF; using fallback dummy document.")
        try:
            from langchain.schema import Document
        except ImportError:
            from langchain.docstore.document import Document
        dummy_doc = Document(page_content="This is a fallback health document. Please replace health.pdf with a valid PDF file.")
        documents = [dummy_doc]

    # Set up the FAISS vector store with embeddings
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    try:
        db = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        raise ValueError(f"Error creating FAISS index: {e}")

    # Set up the retriever
    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    # Define the prompt template (unchanged logic)
    prompt_template = """
You are a helpful assistant who generates answers only from the provided context.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Answer in a single line.

Context: {context}

Question: {question}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the RetrievalQA chain (preserving your original logic)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Initialize the QA system
qa = init_qa_system()

# -------------------- Flask App --------------------
app = Flask(__name__, template_folder='.')

# Landing page (React/Tailwind based)
@app.route("/")
def index():
    return render_template("index.html")

# Chatbot page (React/Tailwind based)
@app.route("/bot")
def bot():
    return render_template("bot.html")

# API endpoint for processing chat queries
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'result': "Please ask a valid question."})

    response = qa(question)
    return jsonify({'result': response["result"] if "result" in response else "I don't know."})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
