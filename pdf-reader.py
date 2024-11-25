import pdfplumber
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from collections import Counter
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

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

def summarize_text(text, max_chunk_length=512):
    # Tokenize the text and split into manageable chunks of max_chunk_length tokens
    summarizer = pipeline("summarization")
    words = nltk.word_tokenize(text)
    chunks = [words[i:i + max_chunk_length] for i in range(0, len(words), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
        # Join words back into a string and summarize
        chunk_text = ' '.join(chunk)
        try:
            summary = summarizer(chunk_text, max_length=200, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append("[Error during summarization]")
    
    # Combine the summaries into a final one
    return " ".join(summaries)

def extract_keywords(text, num_keywords=5):
    # Tokenize and clean the text
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Using TF-IDF Vectorizer for keyword extraction
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
    
    # Get the feature names (keywords)
    keywords = vectorizer.get_feature_names_out()
    
    return keywords

# Replace with your PDF file path
pdf_file = "download.pdf"

# Extract text
paragraphs = extract_main_paragraphs(pdf_file)

# Summarize the text
summary = summarize_text(paragraphs)

# Extract top 5 keywords
keywords = extract_keywords(paragraphs, num_keywords=5)

# Define the output text file path
output_text_file = "extracted_text.txt"

# Save the extracted text and summary to a text file
with open(output_text_file, "w", encoding="utf-8") as f:
    f.write(f"Summary:\n{summary}\n\n")
    f.write(f"Keywords: {', '.join(keywords)}\n\n")
    f.write(f"Full Extracted Text:\n{paragraphs}")

print(f"Text extracted, summarized, and saved to {output_text_file}")
