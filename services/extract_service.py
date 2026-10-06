from pymupdf import pymupdf
from pathlib import Path
from langchain.chat_models import BaseChatModel
from pymupdf.pymupdf import Document
import re
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage

class ExtractService:
    def __init__(self, lm: BaseChatModel):
        self._lm = lm

    def open(self, path: Path) -> Document:
        return pymupdf.open(path)

    def extract(self, doc: Document) -> List[str]:
        all_data = []
        for page in doc:
            texts = page.get_text()
            clean_texts = re.sub(r"\n", " ", texts).strip()

            images = page.get_images()
            tables = page.get_tables()
            xrefs = [image[0] for image in images]
            all_data.append({
                "page": page,
                "text": clean_texts,
                "xrefs": [doc.extract_image(xref) for xref in xrefs],
                "tables": tables
            })
        return all_data

    def describe(self, data: bytes, context_text: str, prompt: str = "Hãy phân tích và mô tả nội dung trang này:") -> str:
        content = [
            {
                "type": "text", 
                "text": f"{prompt}\n\nVăn bản trích xuất được từ PDF:\n{context_text}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{data}"
                }
            }
        ]
        message = HumanMessage(content=content)
        response = self._lm.invoke([message])
        return response.content