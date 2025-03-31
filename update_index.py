import os
import shutil
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- LLM Setup ---
Settings.llm = Ollama(model="mistral")

# --- Embedding Model Setup ---
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# --- Storage Folder ---
storage_dir = "./storage_indexes"
backup_dir = "./backup"

# --- Backup Function ---
def backup_storage():
    if os.path.exists(storage_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"storage_backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copytree(storage_dir, backup_path)
        print(f"=> Backup created at {backup_path}")
    else:
        print("=> No existing storage found. Skipping backup.")

# --- Step 1: Backup ---
backup_storage()

# --- Step 2: Load existing index ---
print("=> Loading existing index...")
storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
index = load_index_from_storage(storage_context)
print("=> Index loaded.")

# --- Step 3: Load NEW documents only ---
print("=> Loading new documents (from ./documents_update)...")
documents = SimpleDirectoryReader(input_dir="./documents_update").load_data()
print(f"=> Loaded {len(documents)} new documents.")

# --- Step 4: Convert Documents to Nodes ---
print("=> Converting documents to nodes...")
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)
print(f"=> Converted to {len(nodes)} nodes.")

# --- Step 5: Insert nodes into the existing index ---
print("=> Inserting nodes into the existing index...")
index.insert_nodes(nodes)

# --- Step 6: Persist the updated index ---
index.storage_context.persist(persist_dir=storage_dir)
print("=> Index has been UPDATED (not overwritten) and saved.")
