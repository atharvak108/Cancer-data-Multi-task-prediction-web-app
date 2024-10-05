import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text data
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    # Tokenize, remove stopwords, and lemmatize
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Function to retrieve relevant text using FAISS index
def retrieve(index, model, query, k=3):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_embedding, k)
    return I, D

# Function to generate answer using a sequence-to-sequence model
def generate_answer(retrieved_texts, query, generator_model, gen_tokenizer):
    context = " ".join(retrieved_texts)
    input_text = f"Context: {context} Query: {query}"
    inputs = gen_tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = generator_model.generate(**inputs, max_length=200)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function for RAG pipeline
def rag_pipeline(query, df, model, index, generator_model, gen_tokenizer, k=3):
    indices, _ = retrieve(index, model, query, k)
    retrieved_texts = df.iloc[indices[0]]['cleaned_transcription'].tolist()
    answer = generate_answer(retrieved_texts, query, generator_model, gen_tokenizer)
    return answer

# Loading models and data for reuse in Streamlit
def load_text_generation_models():
    # Load the dataset and clean transcription (assumes it's already cleaned)
    df = pd.read_csv('cleaned_data.csv')

    # Load pre-built FAISS index
    index = faiss.read_index('sbtf_vectorstore.index')

    # Load Sentence Transformer model (same model used during index creation)
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

    # Load generator model for text generation (e.g., BART model)
    generator_name = "facebook/bart-large-cnn"
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_name)
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_name)

    return df, model, index, generator_model, gen_tokenizer