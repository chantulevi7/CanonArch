import os
from PyPDF2 import PdfReader

def extract_and_store_pdf_pages(pdf_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the PDF file
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    
    for i in range(num_pages):
        # Extract text from each page
        page = reader.pages[i]
        text = page.extract_text()

        text = ' '.join(text.split('\n'))

        if len(text) > 0:
            # Define the output file path for the current page
            output_file_path = os.path.join(output_dir, f'page_{i+1}.txt')
            
            # Write the extracted text to a text file
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            
            print(f'Page {i+1} saved to {output_file_path}')

# pdf_path = "Data\PDFs\Francis.pdf"
# output_dir = "Data\Books\Francis"
# extract_and_store_pdf_pages(pdf_path, output_dir)

# pdf_path = "Data\PDFs\Harry.pdf"
# output_dir = "Data\Books\Harry"
# extract_and_store_pdf_pages(pdf_path, output_dir)

pdf_path = "Data\PDFs\Kenneth.pdf"
output_dir = "Data\Books\Kenneth"
extract_and_store_pdf_pages(pdf_path, output_dir)