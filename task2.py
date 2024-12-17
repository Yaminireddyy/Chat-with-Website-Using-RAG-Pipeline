import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from transformers import pipeline


def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p') 
    text_content = "\n".join([p.get_text() for p in paragraphs])
    
    return text_content
def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [ ' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size) ]
    return chunks

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return embeddings


def create_faiss_index(embeddings, metadata):
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings)) 
    return index

def query_to_embedding(query):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    return query_embedding

def similarity_search(query_embedding, index, k=5):
    distances, indices = index.search(query_embedding.numpy(), k)  
    return indices, distances

def get_relevant_chunks(indices, metadata, k=5):
    relevant_chunks = [metadata[i] for i in indices[0][:k]]  # 'indices[0]' gives the list of indices
    return relevant_chunks


# Load a pre-trained model for text generation (GPT-2 or any suitable model)
generator = pipeline('text-generation', model='gpt2')

def generate_response(query, relevant_chunks):
    context =  " ".join([item["chunk"] for item in relevant_chunks])  # Combine the relevant chunks as context
    prompt = f"Question: {query}\nAnswer based on the following context:\n{context}\nAnswer: "
    response = generator(prompt, max_length=500,max_new_tokens=500,truncation=True)
    return response[0]['generated_text']

url = "https://www.washington.edu/"
content = scrape_website(url)

chunks = chunk_text(content)
embeddings = generate_embeddings(chunks)

metadata = [{"url": url, "chunk": chunk} for chunk in chunks]  # Example metadata
index = create_faiss_index(embeddings, metadata)

query = "to which organization this website belong to"
query_embedding = query_to_embedding(query)

indices, _ = similarity_search(query_embedding, index)

relevant_chunks = get_relevant_chunks(indices, metadata)

response = generate_response(query, relevant_chunks)
print(response)