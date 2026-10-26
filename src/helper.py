
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

#change working directory to the data folder
os.chdir(r"D:\Study\PythonProjects\Medical Chatbot\Medical-Chatbot-")
print(os.getcwd())


def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path / "data", glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()



def filter_data(extracted_data):
    filtered_documents = []

    for doc in extracted_data:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        content = doc.page_content

        filtered_documents.append(
            Document(
                page_content=content,
                metadata={
                    "page": page,
                    "source": source
                }
            )
        )

    return filtered_documents





def split_text(filtered_doc):
    text_splitting=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )
    chunks=text_splitting.split_documents(filtered_doc)
    return chunks



def create_embeddings():

    embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
    
    
    )

    # vectors=[]

    # for chunk in get_chunks:
    #     vectors=embedding_model.embed_documents(chunk.page_content)
    return embedding_model