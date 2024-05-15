import os
import json
import fitz  # PyMuPDF for reading PDF files
import ollama
import numpy as np
import streamlit as st
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_pdf(filename):
    """
    Parse a PDF file and extract paragraphs.
    
    Args:
        filename (str): The name of the PDF file to parse.
        
    Returns:
        list: A list of paragraphs.
    """
    doc = fitz.open(filename)
    paragraphs = []
    for page in doc:
        paragraphs.extend(page.get_text("text").split("\n\n"))
    return paragraphs

def save_embeddings(filename, embeddings):
    """
    Save embeddings to a JSON file.
    
    Args:
        filename (str): The name of the file.
        embeddings (list): A list of embeddings to save.
    """
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    base_filename = os.path.basename(filename)
    with open(f"embeddings/{base_filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    """
    Load embeddings from a JSON file.
    
    Args:
        filename (str): The name of the file.
        
    Returns:
        list or bool: The loaded embeddings or False if the file does not exist.
    """
    base_filename = os.path.basename(filename)
    if not os.path.exists(f"embeddings/{base_filename}.json"):
        return False
    with open(f"embeddings/{base_filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, chunks):
    """
    Get or compute embeddings for the given text chunks.
    
    Args:
        filename (str): The name of the file.
        chunks (list): A list of text chunks to embed.
        
    Returns:
        list: A list of embeddings.
    """
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = embedding_model.encode(chunks).tolist()
    save_embeddings(filename, embeddings)
    return embeddings

def find_similar(needle, haystack):
    """
    Find the most similar embeddings to the given needle embedding.
    
    Args:
        needle (list): The embedding to compare.
        haystack (list): A list of embeddings to search.
        
    Returns:
        list: A sorted list of similarity scores and indices.
    """
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def model_res_gen(prompt, context, model_name):
    """
    Generate responses from the model using streaming.
    
    Args:
        prompt (str): User's prompt.
        context (str): Context from the document.
        model_name (str): The model name.
        
    Yields:
        str: Chunk of the assistant's response.
    """
    stream = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    response_buffer = ""
    for chunk in stream:
        response_buffer += chunk["message"]["content"]
        yield response_buffer

def chat_with_document(file_path):
    """
    Chat with the document by providing responses based on the file content.
    
    Args:
        file_path (str): The path to the file.
    """
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
                       based on the snippets of text provided in context. Answer
                       only using the context provided, being as concise as possible.
                       If you are unsure, just say you don't know.
                       Context:
                       """
    
    paragraphs = parse_pdf(file_path)
    embeddings = get_embeddings(file_path, paragraphs)

    st.title("Document Chatbot")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Model selection
    if "model" not in st.session_state:
        st.session_state["model"] = ""

    models = [model["name"] for model in ollama.list()["models"]]
    st.session_state["model"] = st.selectbox("Choose Model", models)
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])
    
    # User input prompt
    if prompt := st.chat_input("Ask a question about the document"):
        # Generate embedding for the user query
        prompt_embedding = embedding_model.encode(prompt)
        # Find the most similar paragraphs in the document
        most_similar_chunks = find_similar(prompt_embedding, embeddings)[:5]
        
        # Create context from the most similar paragraphs
        context = SYSTEM_PROMPT + "\n".join(paragraphs[item[1]] for item in most_similar_chunks)
        
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
        # Stream and display the assistant's response
            assistant_response = ""
            response_placeholder = st.empty()  # Create a placeholder for the assistant's response
            for chunk in model_res_gen(prompt, context, st.session_state["model"]):
                assistant_response = chunk
                response_placeholder.markdown(assistant_response)  # Update the response
        
        # Append assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

# Streamlit UI for uploading a document
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:
    # Save the uploaded file to a local directory
    file_path = os.path.join("uploads", uploaded_file.name)
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Start the chat with the uploaded document
    chat_with_document(file_path)