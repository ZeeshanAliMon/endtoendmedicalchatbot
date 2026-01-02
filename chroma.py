from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# Load PDFs
loader = DirectoryLoader(
    "./Data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
print("pdf loaded")
# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)
print("chunks made")
print("Total chunks:", len(chunks))

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma()
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Store with progress bar
BATCH_SIZE = 100
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding & storing"):
    vectordb.add_documents(chunks[i:i+BATCH_SIZE])

vectordb.persist()
print("✅ Chroma DB built successfully")
