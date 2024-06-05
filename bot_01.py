# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub
# import os
# from langchain.prompts.prompt import PromptTemplate

# import requests
# from pdfminer.high_level import extract_text

# # Define the PDF file path
# file_path = r'C:\Users\91811\Desktop\chat bot NCS\NCS Dataset.pdf'

# # Extract text from the PDF file
# text = extract_text(file_path)
# ## extracting text from pdf files
#                             # def get_pdf_text(pdf_docs):
#                             #     text = ""
#                             #     for pdf in pdf_docs:
#                             #         pdf_reader = PdfReader(pdf)
#                             #         for page in pdf_reader.pages:
#                             #             text += page.extract_text()
#                             #     return text
# ## creating overlapping text chunks
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks
# ## creating embeddings for chunks of text
# def get_vectorstore(text_chunks):
#     #embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore
# ## ceating a retrival llm chain
# def retrieval_qa_chain(db,return_source_documents):
#     llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.6,"max_length":500, "max_new_tokens":700})
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=db,
#                                        return_source_documents=return_source_documents,
#                                        )
#     return qa_chain
# ## DATA VECTORIZATION AND INDEX CREATION
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#                             # path_to_pdf = ['./content/MyDrive/data_set/new_diseases_data.pdf']
#                                 # raw_text = get_pdf_text(path_to_pdf)
# raw_text = text
# # get the text chunks
# text_chunks = get_text_chunks(raw_text)
# # create vector store
# vectorstore = get_vectorstore(text_chunks)
# ## creating a db with similarity search and obtaining top 3 most matched vectors of all the vectors present in vector index
# db = vectorstore.as_retriever(search_kwargs={'k': 3})
# ## passing database to bot as input and initializing the bot
# bot = retrieval_qa_chain(db,True)
# ## passing query to llm
# query = "what is  Nibble Computer Society?"
# sol=bot(query)
# ## answer giveb by llm
# print(sol['result'])
# # these are the text chunks matched with llm
# print(sol['source_documents'])

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from pdfminer.high_level import extract_text

# Define the PDF file path
file_path = r'C:\Users\91811\Desktop\chat bot NCS\NCS Dataset.pdf'

# Extract text from the PDF file
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
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
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
query = "what is Nibble Computer Society?"
sol = bot(query)

# Answer given by LLM
print(sol['result'])

# These are the text chunks matched with LLM
for doc in sol['source_documents']:
    print(doc.page_content)




# normal falcon without context
# llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.7,"max_length":500, "max_new_tokens":700})
# llm(query)
# ques=['what are the origins od Numpalofich Legatrosis',
#       'what are the stages of diseases progression in Ramtronephiach Oculosis',
#       'what is mortality rate in Wallmic Pulmora',
#       'is Numpalactics incubation period short?',
#       ' what is Numpalactic',
#       ' What are the symptoms of a disease that causes blindness?',
#       'what are the origins of Ramtronephiach Oculosis']
# sol=bot(ques[0])
# print(ques[0])
# print(sol['result'])
# sol=bot(ques[1])
# print(ques[1])
# print(sol['result'])
# sol=bot(ques[2])
# print(ques[2])
# print(sol['result'])
# sol=bot(ques[3])
# print(ques[3])
# print(sol['result'])
# sol=bot(ques[4])
# print(ques[4])
# print(sol['result'])
# sol=bot(ques[5])
# print(ques[5])
# print(sol['result'])
# sol=bot(ques[6])
# print(ques[6])
# print(sol['result'])