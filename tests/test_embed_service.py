from unittest.mock import MagicMock
import pytest
from langchain_huggingface import HuggingFaceEmbeddings
from services.embed_serivce import EmbedService


@pytest.fixture
def embed_model() -> MagicMock:
    model = MagicMock(spec=HuggingFaceEmbeddings)
    model.embed_documents.side_effect = lambda texts: [[0.0, 0.1, 0.2] for _ in texts]
    # 25/9/2026: quên không thêm dòng này vào nên test_embed_query bị lỗi
    model.embed_query.return_value = [0.0, 0.1, 0.2]
    return model


@pytest.fixture
def service() -> EmbedService:
    return EmbedService()


def test_embed_document(service, embed_model):
    texts = ["Hello", "goodbyte"]
    result = service.embed_documents(embed_model, texts)
    assert result == [[0.0, 0.1, 0.2] for _ in range(len(texts))]


def test_embed_query(service, embed_model):
    query = "Hello"
    result = service.embed_query(embed_model, query)
    assert [0.0, 0.1, 0.2] == result
