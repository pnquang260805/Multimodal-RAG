from configs.app_config import AppConfig

from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


class StoreService:
    def __init__(self, embeding_model: HuggingFaceEmbeddings):
        self.conf = AppConfig()
        self.client = MongoClient(self.conf.MONGODB_CONNEC_URL)
        collection_name = self.conf.ATLAS_COLLECTION
        db_name = self.conf.DB_NAME
        collection = self.client[db_name][collection_name]
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=collection, embedding=embeding_model, index_name="default"
        )

    def create_vector_store(
        self, dimentions: int = 1024, wait_until_complete: int = 45
    ):
        self.vector_store.create_vector_search_index(
            dimensions=dimentions, wait_until_complete=wait_until_complete
        )
