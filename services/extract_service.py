from pymupdf import pymupdf
from pathlib import Path
from langchain.chat_models import BaseChatModel
from pymupdf.pymupdf import Document, Page
import re
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
import base64
from langchain_core.documents import Document as LCDocument

class ExtractService:
    def open(self, path: Path) -> Document:
        return pymupdf.open(path)

    def extract(self, path: Path) -> List[Document]:
        docs: List[Document] = []
        with self.open(path) as pdf:
            for page in pdf:
                text = page.get_text()
                text = re.sub(r"[ \t]+", " ", text)       # gộp khoảng trắng
                text = re.sub(r"\n{3,}", "\n\n", text)    # gộp dòng trống thừa
                text = text.strip()
                if not text:                               # bỏ trang rỗng (ảnh scan)
                    continue
                docs.append(
                    LCDocument(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "page": page.number,
                        },
                    )
                )
        return docs

    def extract_dir(self, folder: str) -> List[Document]:
        all_docs: List[Document] = []
        for pdf_path in Path(folder).rglob("*.pdf"):
            all_docs.extend(self.extract(pdf_path))
        return all_docs