
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from helper import load_pdf_files,filter_data,split_text,create_embeddings
import os 
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbottest"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Must match your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully!")
else:
    print("Index already exists!")


index = pc.Index(index_name)

extracted_data=load_pdf_files(Path.cwd())
filtered_documents=filter_data(extracted_data)
chunks=split_text(filtered_documents)
embedding_model=create_embeddings()


vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
   
)

ids = vector_store.add_documents(
    documents=chunks
)

print(f"Successfully stored {len(ids)} chunks in Pinecone!")