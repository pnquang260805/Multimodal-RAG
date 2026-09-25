from unstructured.documents.elements import Element
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


class EmbedService:
    def embed_documents(
        self, embed_model: HuggingFaceEmbeddings, texts: list[str]
    ) -> List[List[float]]:
        return embed_model.embed_documents(texts)

    def embed_query(
        self, embed_model: HuggingFaceEmbeddings, query: str
    ) -> List[float]:
        return embed_model.embed_query(query)
