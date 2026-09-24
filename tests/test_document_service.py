from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage
from unstructured.documents.elements import (
    CompositeElement,
    ElementMetadata,
    Image,
    ListItem,
    NarrativeText,
    Table,
    Title,
)

from services.document_service import DocumentService


def make_chunk(*elements) -> CompositeElement:
    """Tạo 1 chunk giống output của partition_pdf (chứa orig_elements)."""
    return CompositeElement(
        text="",
        metadata=ElementMetadata(orig_elements=list(elements)),
    )


def make_image(b64="AAAA", mime="image/png") -> Image:
    return Image(
        text="",
        metadata=ElementMetadata(image_base64=b64, image_mime_type=mime),
    )


@pytest.fixture
# Fixture là một hàm chuẩn bị sẵn thứ mà test cần (một object, dữ liệu, kết nối, mock...), rồi pytest tự động đưa nó vào test khi test yêu cầu.
# Chữ "fixture" nghĩa gốc là "vật cố định, đồ gá". Hình dung như đồ nghề bày sẵn trên bàn trước khi làm việc: test không tự đi lấy đồ, đồ được đặt sẵn trước mặt.
# Ví dụ đời thường
# Bạn muốn kiểm tra một cái máy pha cà phê. Trước mỗi lần kiểm tra, bạn phải: lấy máy ra, cắm điện, đổ nước vào.
# Ba việc này là chuẩn bị, không phải phần kiểm tra. Fixture chính là phần chuẩn bị đó, tách riêng khỏi phần kiểm tra.
# Nếu không có fixture, mỗi test phải tự viết lại phần chuẩn bị (DocumentService(), MagicMock()...),
# lặp đi lặp lại và khó sửa. Có fixture thì viết một lần, dùng ở mọi test, và mỗi test vẫn nhận một bản mới độc lập với nhau.
def service() -> DocumentService:
    return DocumentService()


@pytest.fixture
def model() -> MagicMock:
    m = MagicMock()
    m.invoke.return_value = AIMessage("This is description")
    return m


def test_create_description_return_str(
    service, model
):  # model trùng tên với hàm model() đã fixture
    result = service._create_desription(model, make_image())
    assert result == "This is description"
    assert isinstance(result, str)


def test_processing_data(service, model):
    chunks = [
        make_chunk(NarrativeText("A")),
        make_chunk(make_image()),
        make_chunk(NarrativeText("B"), NarrativeText("C")),
    ]
    result = service.processing_data(model, chunks)
    expected = ["A", "This is description", "B C"]
    assert result == expected
    assert model.invoke.call_count == 1
