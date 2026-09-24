from unstructured.documents.elements import Element
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from typing import List


class DocumentService:
    def _create_desription(self, model: BaseChatModel, element: Element) -> str:
        # Bảng hay ảnh đều là image b64
        image_base64 = element.metadata.image_base64
        mime = element.metadata.image_mime_type or "image/jpeg"
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for document retrieval. "
                        "If it is a chart or table, summarize the key data and trends."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_base64}"},
                },
            ]
        )
        response = model.invoke([message])
        return str(response.content)

    def processing_data(self, model: BaseChatModel, chunks: List[Element]) -> List[str]:
        res = []
        for chunk in chunks:
            temp = ""
            for ele in chunk.metadata.orig_elements:
                ele_dict = ele.to_dict()
                tag = ele_dict.get("type")
                if "Text" in tag:
                    temp += ele_dict.get("text") + " "
                elif "Image" in tag or "Table" in tag:
                    description = self._create_desription(model, ele)
                    temp += description + " "
            res.append(temp.strip())
        return res
