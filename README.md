# PDF Question Answering System

### Overview
PDF Question Answering System is an AI-powered application where users can upload PDF documents and ask questions related to the content of the uploaded PDFs. The system leverages **Llama 3.1** for language understanding and **Weaviate Vector Database** to efficiently store and retrieve relevant document chunks.

### Features
- Upload PDF documents
- Automatic text extraction from PDFs
- Chunking of documents for better search results
- Semantic search using Weaviate Vector Database
- Question Answering using Llama 3.1
- Interactive Web Interface using Streamlit

### Technologies Used
- **Llama 3.1**: Language model for question answering
- **Weaviate Vector DB**: Storing and retrieving document chunks
- **Streamlit**: Web application framework
- **Python**: Backend Development

### Folder Structure
```
├── uploaded_pdfs       # Folder to store uploaded PDFs
├── .gitignore          # Ignore unnecessary files
├── .env                # Environment variables (API keys, DB credentials, etc.)
├── QA_from_uploaded_pdf.ipynb # Jupyter Notebook for testing
├── app.py              # Main Streamlit app
├── requirements.txt    # List of dependencies
└── README.md           # Project Documentation
```

### Installation
1. Clone the repository:
```bash
https://github.com/Prabesh1sharma/PDF-Question-Answering.git
cd PDF-Question-Answering
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add the following:
```plaintext
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API=your_weaviate_api_key
GROQ_API_KEY=your_groq_api_key
```

### How to Run the App
1. Start the Streamlit app:
```bash
streamlit run app.py
```
2. Upload your PDF and ask any question related to its content.
