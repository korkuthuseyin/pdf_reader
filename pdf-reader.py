import pdfplumber

def extract_main_paragraphs(pdf_path):
    main_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text from the page
            text = page.extract_text()

            # Skip pages without text
            if not text:
                continue

            # Filter main text by ignoring headers or side texts
            lines = text.split('\n')
            for line in lines:
                # Basic filtering: Skip short lines or lines with all caps (common in headers)
                if len(line) > 50 and not line.isupper():
                    main_text.append(line)

    return "\n".join(main_text)

# Replace with your PDF file path
pdf_file = "download.pdf"

# Extract text
paragraphs = extract_main_paragraphs(pdf_file)

# Define the output text file path
output_text_file = "extracted_text.txt"

# Save the extracted text to a text file
with open(output_text_file, "w", encoding="utf-8") as f:
    f.write(paragraphs)

print(f"Text extracted and saved to {output_text_file}")
