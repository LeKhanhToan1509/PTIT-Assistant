from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from helper.clean_text import clean_text
dotenv.load_dotenv()

class PdfAction:
    def __init__(self, directory_path = "", path = "Data", api_key = None):
        self.directory_path = directory_path
        self.loaders = []
        self.path = path
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

    def create_vector_database(self):
        folders = os.listdir(self.path)
        loaders = []

        for folder in folders:
            path_folder = os.path.join(self.path, folder)
            loader = DirectoryLoader(
                path_folder,
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True,
            )
            loaders.extend(loader.load())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30, separators=["\n\n", "\n", "\r\n", "\r"])
            chunks = text_splitter.split_documents(loaders)
            embeddings = OpenAIEmbeddings(api_key=self.api_key)
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(self.directory_path)  

    def extend_vector_database(self):
        db = FAISS.load_local(self.directory_path)
        
        folders = os.listdir(self.path)
        new_loaders = []

        for folder in folders:
            path_folder = os.path.join(self.path, folder)
            loader = DirectoryLoader(
                path_folder,
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True,
            )
            new_loaders.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30, separators=["\n\n", "\n", "\r\n", "\r"])
        new_chunks = text_splitter.split_documents(new_loaders)
        
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        new_db = FAISS.from_documents(new_chunks, embeddings)
        
        db.add_documents(new_db.get_documents())
        
        db.save_local(self.directory_path)
    def load_vector_database(self):
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        db = FAISS.load_local(self.directory_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        return db

    def get_context(self, question = "", k=2):
        db = self.load_vector_database()
        result = db.search(question, k=k, search_type='similarity')
        context = " ".join([doc.page_content for doc in result])
        context = clean_text(context)
        return context