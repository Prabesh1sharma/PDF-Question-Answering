from langchain.vectorstores import Weaviate
import weaviate
import locale
locale.getpreferredencoding = lambda: "UTF-8"
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import asyncio
from cachetools import TTLCache
import nest_asyncio
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration for Weaviate and API keys
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API = os.getenv("WEAVIATE_API")

# Initialize Weaviate client
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API)
)

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
llm = ChatGroq(
    temperature=0,
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model="llama-3.1-70b-versatile"
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

# Define a prompt template for question answering
from langchain.prompts import ChatPromptTemplate
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Initialize the cache for vector DB and answers
vector_db_cache = {}
cache = TTLCache(maxsize=100, ttl=600)

# Function to load and process PDF only once
def load_and_process_pdf(pdf_path):
    # Check if the PDF is already processed
    if pdf_path in vector_db_cache:
        return vector_db_cache[pdf_path]
    
    # Load and split PDF content into documents
    loader = PyPDFLoader(pdf_path, extract_images=True)
    pages = loader.load()
    docs = text_splitter.split_documents(pages)
    
    # Insert documents into the vector DB
    vector_db = Weaviate.from_documents(
        docs, embeddings, client=client, by_text=False
    )
    
    # Cache the vector DB for future queries
    vector_db_cache[pdf_path] = vector_db
    
    return vector_db

# Function to handle question-answering from the cached vector DB
def answer_question_from_pdf(pdf_path, question):
    # Check if the answer is already cached
    if question in cache:
        return cache[question]

    # Retrieve vector DB from cache
    vector_db = vector_db_cache.get(pdf_path)

    if not vector_db:
        raise ValueError("Vector database not found. Please upload and process a PDF first.")

    # Set up retrieval and RAG chain
    retriever = vector_db.as_retriever()
    output_parser = StrOutputParser()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    
    # Get the answer and store in cache
    ans = rag_chain.invoke(question)
    cache[question] = ans
    
    return ans

# Streamlit UI setup
st.title("📄 PDF Question-Answering System")
st.sidebar.header("Upload PDF")

# Sidebar for PDF upload
uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Process the uploaded PDF if provided
pdf_path = None
if uploaded_pdf:
    # Save the uploaded PDF to a temporary directory
    temp_dir = "uploaded_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

    # Write the contents of the uploaded file to the new file
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # Process the PDF and store in vector_db if not already done
    if pdf_path not in vector_db_cache:
        with st.spinner("Loading PDF and processing..."):
            load_and_process_pdf(pdf_path)

    st.success("PDF loaded successfully! You can now ask questions.")
else:
    st.warning("Please upload a PDF to start.")

# Input for question and answer display
if pdf_path:
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            # Display only "Finding the answer..." during question-answering
            with st.spinner("Finding the answer..."):
                answer = answer_question_from_pdf(pdf_path, question)
            st.write("**Answer:**", answer)
        else:
            st.warning("Please enter a question.")

# Customize the Streamlit UI
st.markdown("""
<style>
    .stSidebar { 
        background-color: #454545;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
