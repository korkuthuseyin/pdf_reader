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

special_words = ["ben", "sen", "o", "biz", "siz", "onlar", "kendi", "bu", "şu", "ne", 
    "ne kadar", "her", "hiç", "çok", "az", "bu kadar", "şimdi", "geçmiş", 
    "şey", "kim", "ne", "için", "herkes", "gibi", "gerek", "ama", "de", 
    "da", "çünkü", "ya", "ve", "veya", "ile", "ama", "fakat", "ya da", "şimdi", 
    "geçmiş", "şunu", "bunu", "için", "gibi", "nasıl", "nerede",
    "ve", "ile", "ama", "fakat", "ya", "ya da", "ne", "ne kadar", 
    "her", "hiç", "şu", "bu", "diye", "çünkü", "için", "de", "da", 
    "ama", "gerek", "olarak", "önce", "sonra", "kadar", "hem", "ya da", 
    "üzerine", "gibi", "fakat", "ancak", "şimdi", "şöyle", "bu yüzden"]

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

def extract_keywords(text, num_keywords=5, special_words=None):
    # Tokenize and clean the text
    words = nltk.word_tokenize(text.lower())  # Tokenizing the text
    words = [word for word in words if word.isalnum()]  # Remove non-alphanumeric words
    
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))  # Set of stopwords in English
    filtered_words = [word for word in words if word not in stop_words]
    
    # If special_words list is not provided, use an empty list
    if special_words is None:
        special_words = []

    # Filter out the special words (those that should not be added)
    filtered_words = [word for word in filtered_words if word not in special_words]

    # Using TF-IDF Vectorizer for keyword extraction
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])  # Fit the model to the filtered words
    
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
keywords = extract_keywords(paragraphs, num_keywords=5, special_words=special_words)

# Define the output text file path
output_text_file = "extracted_text.txt"

# Save the extracted text and summary to a text file
with open(output_text_file, "w", encoding="utf-8") as f:
    f.write(f"Summary:\n{summary}\n\n")
    f.write(f"Keywords: {', '.join(keywords)}\n\n")
    f.write(f"Full Extracted Text:\n{paragraphs}")

print(f"Text extracted, summarized, and saved to {output_text_file}")
