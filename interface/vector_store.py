from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        pass

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> List[str]:
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        pass


    @abstractmethod
    def _create_retriever(self, search_type: str = "similarity", top_k: int = 5) -> None:
        pass

    @abstractmethod
    def _create_vector_store(
        self,
    ) -> None:
        pass
