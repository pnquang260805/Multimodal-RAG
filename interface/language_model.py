from langchain_core.prompts import ChatPromptTemplate
from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chat_models import BaseChatModel


class LanguageModel(ABC):
    @abstractmethod
    def create_model(self) -> BaseChatModel:
        pass

    @abstractmethod
    def _build_system_prompt(self) -> ChatPromptTemplate:
        pass

    @abstractmethod
    def run_chain(
        self, question: str, prompt: ChatPromptTemplate, retriever: VectorStoreRetriever
    ) -> str:
        pass
