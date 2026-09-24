from unittest.mock import MagicMock
import pytest
from langchain_huggingface import HuggingFaceEmbeddings
from services.embed_serivce import EmbedService


@pytest.fixture
def embed_model() -> MagicMock:
    model = MagicMock(spec=HuggingFaceEmbeddings)
    model.embed_documents.side_effect = lambda texts: [[0.0, 0.1, 0.2] for _ in texts]
    return model


@pytest.fixture
def service() -> EmbedService:
    return EmbedService()


def test_embed(service, embed_model):
    texts = ["Hello", "goodbyte"]
    result = service.embed(embed_model, texts)
    assert result == [[0.0, 0.1, 0.2] for _ in range(len(texts))]
