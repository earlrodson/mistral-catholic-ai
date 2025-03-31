from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load theology documents from the 'documents' folder
documents = SimpleDirectoryReader(input_dir="./documents_base").load_data()

if not documents:
    raise ValueError("No documents found in the 'documents' folder. Add theology-related text files or PDFs.")

# Use a strong embedding model
# embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-large-v2")  #old version but working,
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  

# Create a vector index from the documents
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Save the index for future use
index.storage_context.persist(persist_dir="./storage_indexes")

print("Vector database created successfully.")
