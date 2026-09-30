from services.store_service import MongoAtalasStoreService
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


@pytest.fixture
def mock_embed():
    mock = MagicMock(spec=HuggingFaceEmbeddings)
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def mock_app_config():
    """Mock AppConfig để trả về các biến môi trường giả lập."""
    with patch("services.store_service.AppConfig") as MockConfig:
        instance = MockConfig.return_value
        instance.MONGODB_CONNECT_URL = "mongodb://mock-uri:27017"
        instance.ATLAS_COLLECTION = "mock_collection"
        instance.DB_NAME = "mock_db"
        instance.MONGO_INDEX = "vector_index"
        yield instance


@pytest.fixture
def service(mock_app_config, mock_embed):
    """Khởi tạo service với MongoClient được mock."""
    with patch("services.store_service.MongoClient") as mock_mongo:
        svc = MongoAtalasStoreService(embedding_model=mock_embed)
        yield svc


@pytest.fixture
def mock_results(service):
    expected = [Document(page_content="Test result")]
    service.vector_store = MagicMock()
    service.vector_store.similarity_search.return_value = expected
    return expected


def test_init(service, mock_app_config, mock_embed):
    assert service.collection_name == "mock_collection"
    assert service.db_name == "mock_db"
    assert service.index_name == "vector_index"
    assert service.embedding == mock_embed


def test_create_collection(service):
    """Kiểm tra việc truy cập vào collection (sửa lỗi self.db -> self.db_name)."""
    service.db_name = "mock_db"
    service.create_collection()
    service.client.__getitem__.assert_called_with("mock_db")


@patch("services.store_service.MongoDBAtlasVectorSearch.from_connection_string")
def test_create_vector_store(mock_from_conn, service):
    mock_vs_instance = MagicMock()
    mock_from_conn.return_value = mock_vs_instance

    service.create_vector_store()

    mock_from_conn.assert_called_once_with(
        connection_string=service.conf.MONGODB_CONNECT_URL,
        namespace="mock_db.mock_collection",
        embedding=service.embedding,
        index_name="vector_index",
    )
    assert service.vector_store == mock_vs_instance


def test_similarity_search(mock_results, service):
    results = service.similarity_search(query="hỏi đáp", top_k=3)
    service.vector_store.similarity_search.assert_called_once_with("hỏi đáp", 3)
    assert results == mock_results
    assert len(results) == 1
    assert results[0].page_content == "Test result"


def test_add_document(mock_results, service):
    res = service.add_documents(mock_results)
    service.vector_store.add_documents.assert_called_once()
