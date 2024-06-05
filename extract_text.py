# import PyPDF2

# # Open the PDF file
# file_path = r'C:\Users\91811\Desktop\chat bot NCS\NCS Dataset.pdf'
# pdf_file = open(file_path, 'rb')

# # Create a PDF reader object using PdfReader
# pdf_reader = PyPDF2.PdfReader(pdf_file)

# # Get the number of pages
# num_pages = len(pdf_reader.pages)

# # Extract text from each page
# for page_num in range(num_pages):
#     page = pdf_reader.pages[page_num]
#     text = page.extract_text()
#     print(f"Page {page_num + 1}:")
#     print(text)

# Close the PDF file
# pdf_file.close()


from pdfminer.high_level import extract_text

# Define the PDF file path
file_path = r'C:\Users\91811\Desktop\chat bot NCS\NCS Dataset.pdf'

# Extract text from the PDF file
text = extract_text(file_path)

# Print the extracted text
print(text)
