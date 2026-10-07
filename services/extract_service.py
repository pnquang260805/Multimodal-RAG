from pymupdf import pymupdf
from pathlib import Path
from langchain.chat_models import BaseChatModel
from pymupdf.pymupdf import Document
import re
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
import base64


class ExtractService:
    def __init__(self, lm: BaseChatModel):
        self._lm = lm
        self.prompt = """
        Hãy tạo description cho ảnh sau và dựa vào context đã cho. 
        Chỉ trả về description của hình ảnh.
        Không trả lời lan man.
        Trả lời ngắn gọn
        """

    def open(self, path: Path) -> Document:
        return pymupdf.open(path)

    def extract(self, doc: Document) -> List[str]:
        all_data = []
        for page in doc:
            texts = page.get_text()
            clean_texts = re.sub(r"\n", " ", texts).strip()

            images = page.get_images()
            tables = page.find_tables()
            xrefs = [img[0] for img in page.get_images(full=True)]
            all_data.append(
                {"page": page, "text": clean_texts, "xrefs": xrefs, "tables": tables}
            )
        return all_data

    def describe(self, data: bytes, context_text: str) -> str:
        data = base64.b64encode(data).decode("utf-8")
        content = [
            {
                "type": "text",
                "text": f"{self.prompt}\n\nVăn bản trích xuất được từ PDF:\n{context_text}",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"},
            },
        ]
        message = HumanMessage(content=content)
        response = self._lm.invoke([message])
        return response.content
