"""Utilities for splitting PDF documents into structured content chunks.

This module provides a thin wrapper around the unstructured PDF partitioning
library so the project can convert document pages and elements into chunkable
segments for downstream retrieval or LLM processing.
"""

from unstructured.partition.pdf import partition_pdf
from typing import List, IO
from unstructured.documents.elements import Element


class ChunkingService:
    def chunking(
        self,
        file: str | IO[bytes],
        strategy: str = "hi_res",
        chunking_strategy: str = "basic",
        languages: List[str] = ["eng"],
    ) -> List[Element]:
        """Split a PDF document into structured chunks.

        Args:
            file: Path to the PDF file or a binary file-like object containing
                the PDF content.
            strategy: PDF extraction strategy used by unstructured, such as
                "hi_res" for higher-quality OCR/layout parsing.
            chunking_strategy: Chunking mode to apply after extraction, for
                example "basic" or "by_title".
            languages: List of languages to use for OCR or text extraction,
                defaulting to English ("eng").

        Returns:
            A list of extracted document elements/chunks produced by
            unstructured.partition_pdf.
        """
        chunks = partition_pdf(  # là bước chunking luôn
            file,  # Hoặc file path str hoặc file IO[Bytes]
            infer_table_structure=True,
            strategy=strategy,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy=chunking_strategy,
            languages=languages,
            extract_images_in_pdf=True,
        )
        return chunks
