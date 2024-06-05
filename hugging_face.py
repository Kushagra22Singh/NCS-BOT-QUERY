# import requests

# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ",
# })

# print(output)


# too large to be loaded


###########################################################
#              FIRST WORKING 
###########################################################


import requests
from pdfminer.high_level import extract_text

# Define the PDF file path
file_path = r'C:\Users\91811\Desktop\chat bot NCS\NCS Dataset.pdf'

# Extract text from the PDF file
text = extract_text(file_path)

API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-cased-distilled-squad"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
		"question": "what is JSS Infotech  ?",
		"context": text
	},
})

print(output)

###########################################################
#              SECOND WORKING (BUT NOT AS GOOD AS ABOVE ONE)
###########################################################

# import requests
print('**********************************************************************************')

API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
		"question": "what is JSS Infotech  ?",
		"context": text
	},
})

print(output)

###########################################################
#              Third not working (on a custom dataset)
###########################################################

# # With pipeline, just specify the task and the model id from the Hub.
# from transformers import pipeline
# pipe = pipeline("text-generation", model="distilbert/distilgpt2")

# # If you want more control, you will need to define the tokenizer and model.
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# # Add a new padding token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))

# # Define your question
# question = "What is the capital of France?"

# # Tokenize the input
# inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

# # Generate a response with attention mask
# outputs = model.generate(
#     input_ids=inputs['input_ids'],
#     attention_mask=inputs['attention_mask'],
#     max_length=50,
#     num_return_sequences=1,
#     pad_token_id=tokenizer.pad_token_id  # Ensure padding uses the new pad token
# )

# # Decode the output
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Print the response
# print(response)









# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# # Set the pad token as the eos token
# tokenizer.pad_token = tokenizer.eos_token

# # Define the question with a structured prompt
# question = "Q: What is the capital of France?\nA:"

# # Tokenize the input
# inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

# # Generate a response with sampling
# outputs = model.generate(
#     input_ids=inputs['input_ids'],
#     attention_mask=inputs['attention_mask'],
#     max_length=50,
#     num_return_sequences=1,
#     pad_token_id=tokenizer.eos_token_id,  # Ensure padding uses the end-of-sequence token
#     do_sample=True,  # Enable sampling
#     temperature=0.7,  # Control the randomness of predictions
#     top_k=50,  # Consider only the top 50 tokens by probability
#     top_p=0.95  # Nucleus sampling: consider tokens with cumulative probability >= 0.95
# )

# # Decode the output
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Print the response
# print(response)


###########################################################
#              Fourth (same as first one)
###########################################################


# import requests
print('**********************************************************************************')

API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
		"question": "what is JSS Infotech  ?",
		"context": text
	},
})

print(output)

###########################################################
#              FIFTH 
###########################################################
print('**********************************************************************************')
from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define the context and question
context = text
question = 'what is JSS Infotech ?'

# Get the answer
result = qa_pipeline(question=question, context=context)
print(f"Q: {question}\nA: {result['answer']}")
