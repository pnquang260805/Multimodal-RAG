from unstructured.documents.elements import Element
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


class EmbedService:
    def embed(
        self, embed_model: HuggingFaceEmbeddings, texts: list[str]
    ) -> List[List[float]]:
        return embed_model.embed_documents(texts)
