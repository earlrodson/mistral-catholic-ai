from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, PromptTemplate, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# --- LLM Setup ---
Settings.llm = Ollama(model="mistral")

# --- Embedding Model Setup (LOCAL and OFFLINE) ---
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# --- Prompt Setup ---
instruction_prompt = PromptTemplate(
   "You are a Catholic Apologist AI. Your goal is to help others understand the beauty and wisdom of the Catholic faith. Always respond with:\n"
    "1. Clear Catholic teaching based on the Catechism and Church documents.\n"
    "2. At least one supporting Bible verse when possible.\n"
    "3. A simple and relatable analogy for better understanding.\n"
    "4. Empathetic language to show understanding of the questioner's concerns or doubts.\n"
    "5. Never invent information, and always stay faithful to Catholic doctrine.\n"
    "6. If the question is not related to Catholicism, politely inform them.\n"
    "7. you provide the bible verse and the catechism reference\n"
    "\n"
    "Conversation history: {conversation_history}\n"
    "Question: {query_str}\n"
    "Answer: Remember to answer based on Catholic teaching, especially on sensitive topics such as suffering, God's existence, and faith.\n"
)

Settings.prompt_template = instruction_prompt

# --- Storage Folder ---
storage_dir = "./storage"

# --- Load Existing Index ---
print("=> Loading existing index from 'storage_indexes/'...")
storage_context = StorageContext.from_defaults(persist_dir=storage_dir)

# Load existing index
index = load_index_from_storage(storage_context)
print("=> Index loaded.")

# --- Query Loop with Session Memory ---
query_engine = index.as_query_engine()

conversation_history = ""  # Store conversation history

while True:
    question = input("Ask a question about Catholicism (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    
    # Build prompt with the conversation history
    prompt = instruction_prompt.format(conversation_history=conversation_history, query_str=question)
    
    # Get response from the query engine
    response = query_engine.query(question)

    # Print answer
    print(f"\nAnswer: {response}\n")
    
    # Add the question and answer to the conversation history
    conversation_history += f"Q: {question}\nA: {response}\n"
