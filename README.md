# PDF Text Extractor and Summarizer

A Python tool that extracts text from PDF files, summarizes the content, and identifies key keywords. This project combines `pdfplumber` for PDF text extraction, `transformers` for text summarization, and `scikit-learn` for keyword extraction, making it ideal for processing and analyzing textual data from documents.

---

## Features

- **Extract Text**: Filters main content while skipping headers, side texts, or irrelevant sections.
- **Summarize Content**: Provides concise summaries of the extracted text using a pre-trained Hugging Face summarization model.
- **Keyword Extraction**: Identifies the top keywords from the content using TF-IDF.
- **Output to File**: Saves the summary, keywords, and full text to a readable `.txt` file.

---

## Installation

To use this tool, ensure you have Python 3.7+ installed and then follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
2. Install required dependencies:
   ```pip install -r requirements.txt```
3. Download NLTK resources:
4. ```bash
   python
   >>> import nltk
   >>> nltk.download('punkt')
   >>> nltk.download('stopwords')

## Usage

1. **Place the PDF in the Project Directory**  
   Update the `pdf_file` variable in the script with your PDF's filename:
   ```python
   pdf_file = "your_pdf_file.pdf"
2. Run the Script
   Execute the script with the following command:
   
   ```bash
   Copy code
   python script.py
3. Output File
   The extracted text, summary, and keywords will be saved in extracted_text.txt in the project directory.


## Dependencies

The following libraries are used in this project:

- **`pdfplumber`**: For PDF text extraction.
- **`transformers`**: For text summarization.
- **`scikit-learn`**: For keyword extraction using TF-IDF.
- **`nltk`**: For natural language processing tasks like tokenization and stop word removal.

Install them with:
```bash
pip install pdfplumber transformers scikit-learn nltk
```

## Example o

After running the script, the `extracted_text.txt` file will contain the following sections:

1. **Summary**: A concise overview of the extracted PDF content.
2. **Keywords**: Top five keywords derived from the text using TF-IDF.
3. **Full Extracted Text**: The entire filtered content extracted from the PDF.

---

## Example Content

```plaintext
Summary:
<Summary of the extracted text>

Keywords:
<keyword1>, <keyword2>, <keyword3>, <keyword4>, <keyword5>

Full Extracted Text:
<Full extracted paragraphs from the PDF>
