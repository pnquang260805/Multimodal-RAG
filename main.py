from services.qwen_service import QwenService
from services.store_service import MongoAtalasStoreService
from configs.app_config import AppConfig
from langchain_huggingface import HuggingFaceEmbeddings
from services.extract_service import ExtractService
conf = AppConfig()
from pprint import pprint

# embeddings = HuggingFaceEmbeddings(
#     model_name=conf.EMB_MODEL_ID,
#     model_kwargs={
#         "device": "cuda",  # hoặc "cpu"
#         "processor_kwargs": {"padding_side": "left"},
#     },
#     encode_kwargs={"normalize_embeddings": True},
# )

# print("====================Vector store======================")
# m = MongoAtalasStoreService(embeddings)
# m.create_collection()
# m.create_vector_store()
# m.create_retriever()
# retriever = m.get_retriever()
# print("======================================================\n\n\n")
print("====================Model======================")
q = QwenService()
q.create_model()
# res = q.run_chain("Tìm cho tôi luật lao động ở nhật", retriever)
# print(res)
print("======================================================\n\n\n")
model = q.lm
e = ExtractService(model)
from pymupdf import pymupdf
doc = e.open(r"E:\Mine\Code\LangchainProject\pdfs\mynumberriyou.pdf")
all_data = e.extract(doc)
pprint(len(all_data))
d = all_data[1]
xrefs = d['xrefs']
text = d['text']
img = doc.extract_image(xrefs[0])['image']
response = e.describe(img, text)
pprint(f"Response: {response}")
