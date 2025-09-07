import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter,TokenTextSplitter
# --- NEW IMPORT ---
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- IMPORTANT CHANGE HERE: Set HF_HOME environment variable ---
# Define the path for the Hugging Face cache directory
hf_cache_directory = "D:\\KELLA BENNANI OTHMANE\\Othmane\\models"
#os.makedirs(hf_cache_directory, exist_ok=True)
os.environ['HF_HOME'] = hf_cache_directory
print(f"Hugging Face cache directory set to: {os.environ['HF_HOME']}")

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "langchain_demo.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_langchain")



# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file with correct encoding
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split the document into chunks
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)#chaque chunk a 1000 caractères et chunk_overlap=0 ceci permet que chaque chunk est independant pas de répetitions des mots
    tokens_splitter = TokenTextSplitter(chunk_size=240, chunk_overlap=20)
    """
        Chunk 1 : ABCDE
        Chunk 2 : FGHIJ
        
        avec overlap = 2 
            Chunk 1 : ABCDE
            Chunk 2 : CDEFG -> répétition de qlq letteres dans les deux chunks
    """ 
    docs = tokens_splitter.split_documents(documents)

    # Create embeddings
    print("\n--- Creating embeddings ---")
    # Initialize the SentenceTransformer model
    #By default, input text longer than 256 word pieces is truncated. (supprimés)
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    sentence_transformer_model = SentenceTransformer(model_name)

    # --- WRAP THE SentenceTransformer MODEL WITH HuggingFaceEmbeddings ---
    embeddings = HuggingFaceEmbeddings(client=sentence_transformer_model) # Use 'client' for an already loaded model
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store ---")
else:
    print("Persistent directory already exists. Loading vector store...")