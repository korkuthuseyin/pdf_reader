import pdfplumber
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from nltk.corpus import stopwords
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

# Function to extract main paragraphs from the PDF
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

# Function to summarize text using TF-IDF (simple rule-based summarization)
def simple_summary(text, num_sentences=5):
    sentences = nltk.sent_tokenize(text)

    # Vectorize the sentences
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Sum up the TF-IDF scores for each sentence
    sentence_scores = tfidf_matrix.sum(axis=1)

    # Get the indices of the top N sentences
    ranked_sentences = sentence_scores.argsort().flatten()[::-1]

    # Select the top N sentences
    summary = ' '.join([sentences[i] for i in ranked_sentences[:num_sentences]])

    return summary

# Function to extract keywords using TF-IDF
def extract_keywords(text, num_keywords=5):
    # Tokenize and clean the text
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    
    # Remove stop words (using Turkish stopwords if available)
    stop_words = set(stopwords.words('english'))  # Change to 'turkish' for Turkish stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Using TF-IDF Vectorizer for keyword extraction
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
    
    # Get the feature names (keywords)
    keywords = vectorizer.get_feature_names_out()
    
    return keywords

# Function to extract keywords using RAKE (Rapid Automatic Keyword Extraction)
def extract_keywords_rake(text):
    rake = Rake(stopwords=nltk.corpus.stopwords.words('english'))  # Change to 'turkish' if needed
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

# Main function to run the process
def main(pdf_file):
    # Extract text from the PDF
    paragraphs = extract_main_paragraphs(pdf_file)
    
    # Summarize the text using simple TF-IDF based summary
    summary = simple_summary(paragraphs, num_sentences=5)
    
    # Extract keywords using TF-IDF
    tfidf_keywords = extract_keywords(paragraphs, num_keywords=5)
    
    # Extract keywords using RAKE
    rake_keywords = extract_keywords_rake(paragraphs)
    
    # Define the output text file path
    output_text_file = "extracted_text.txt"
    
    # Save the extracted text, summary, and keywords to a text file
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(f"Summary:\n{summary}\n\n")
        f.write(f"TF-IDF Keywords: {', '.join(tfidf_keywords)}\n\n")
        f.write(f"RAKE Keywords: {', '.join(rake_keywords)}\n\n")
        f.write(f"Full Extracted Text:\n{paragraphs}")
    
    print(f"Text extracted, summarized, and saved to {output_text_file}")

# Replace with your PDF file path
pdf_file = "download.pdf"

# Run the main process
main(pdf_file)
