import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List

class EmbeddingManager:
    """Classe para gerenciar embeddings e VectorStore com Chroma."""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.persist_directory = "chroma_db"

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Cria um VectorStore Chroma a partir de documentos e salva localmente."""
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
       
        return vector_store

    def load_vector_store(self) -> Chroma:
        """Carrega o VectorStore Chroma do disco, se existir."""
        if os.path.exists(self.persist_directory+"/chroma.sqlite3"):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            raise FileNotFoundError("Nenhum VectorStore salvo encontrado no caminho especificado.")
        
    def get_or_create_vector_store(self, documents: List[Document]) -> Chroma:
        """Verifica se existe um VectorStore, se existir carrega, senão cria um novo."""
        try:
            vector_store = self.load_vector_store()
            print("VectorStore existente carregado com sucesso")
            return vector_store
        except FileNotFoundError:
            print("Criando novo VectorStore")
            return self.create_vector_store(documents)