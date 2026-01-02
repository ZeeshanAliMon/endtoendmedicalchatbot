
from src.helper import download_hugging_face_embeddings,text_split,load_pdf_file 
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as pcone
import os 
load_dotenv()
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
print(PINECONE_API_KEY)
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws",region="us-east-1")
)

docsearch = pcone.from_documents(
    documents=text_chunks,
    index_name= index_name,
    embedding = embeddings
)
