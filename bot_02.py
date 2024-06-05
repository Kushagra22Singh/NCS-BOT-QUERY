import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer

# Define the PDF file path
file_path = r'C:\Users\91811\Desktop\chat bot NCS\NCS Dataset.pdf'

text = extract_text(file_path)

# Create overlapping text chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create embeddings for chunks of text
def get_vectorstore(text_chunks):
    model = SentenceTransformer("hkunlp/instructor-xl")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Create a retrieval LLM chain
def retrieval_qa_chain(db, return_source_documents):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6, "max_length": 500, "max_new_tokens": 700})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db,
                                           return_source_documents=return_source_documents)
    return qa_chain

# Ensure HuggingFace API token is correctly set
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Extract raw text from the PDF
raw_text = text

# Get the text chunks
text_chunks = get_text_chunks(raw_text)

# Create vector store
vectorstore = get_vectorstore(text_chunks)

# Create a database with similarity search
db = vectorstore.as_retriever(search_kwargs={'k': 3})

# Initialize the bot
bot = retrieval_qa_chain(db, True)

# Pass a query to LLM
query = "What is Nibble Computer Society?"
sol = bot(query)

# Answer given by LLM
print(sol['result'])

# These are the text chunks matched with LLM
for doc in sol['source_documents']:
    print(doc.page_content)
