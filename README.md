
---

# NCS Chatbot

This project implements a chatbot using the `langchain` and `sentence-transformers` libraries to interact with a dataset extracted from a PDF document. The bot leverages FAISS for vector similarity search and uses the `falcon-7b-instruct` model for natural language understanding. 

### Features
- Extracts text from PDF files.
- Splits text into manageable chunks for processing.
- Creates embeddings using the `hkunlp/instructor-xl` model.
- Provides accurate responses based on the extracted information.

### Usage
Ensure you have the required dependencies installed and set your HuggingFace API token before running the script.

---
 Note : currently hugging_face.py is working but the other 2 bots namely bot_01.py and bot_02.py are not tested.
