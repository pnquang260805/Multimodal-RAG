from configs.app_config import AppConfig

from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from interface.vector_store import VectorStore
from uuid import uuid4


class MongoAtalasStoreService(VectorStore):
    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self.conf = AppConfig()
        self.client = MongoClient(self.conf.MONGODB_CONNECT_URL)
        self.collection_name = self.conf.ATLAS_COLLECTION
        self.db_name = self.conf.DB_NAME
        self.index_name = self.conf.MONGO_INDEX
        self.embedding = embedding_model

    def create_collection(self):
        self.client[self.db_name][self.collection_name]

    def create_vector_store(self):
        self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=self.conf.MONGODB_CONNECT_URL,
            namespace=f"{self.db_name}.{self.collection_name}",
            embedding=self.embedding,
            index_name=self.index_name,
        )

    def similarity_search(self, query: str, top_k=5):
        return self.vector_store.similarity_search(query, top_k)

    def add_documents(self, docs):
        ids = [str(uuid4()) for _ in range(len(docs))]
        return self.vector_store.add_documents(docs, ids)

    def delete_documents(self, ids):
        return self.vector_store.delete(ids)

    def create_retriever(self, search_type="similarity", top_k=5):
        self.retriever = self.vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": top_k}
        )

    def get_retriever(self):
        return self.get_retriever
