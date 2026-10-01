from services.qwen_service import QwenService
from services.store_service import MongoAtalasStoreService
from configs.app_config import AppConfig
from langchain_huggingface import HuggingFaceEmbeddings

conf = AppConfig()

embeddings = HuggingFaceEmbeddings(
    model_name=conf.EMB_MODEL_ID,
    model_kwargs={
        "device": "cuda",  # hoặc "cpu"
        "processor_kwargs": {"padding_side": "left"},
    },
    encode_kwargs={"normalize_embeddings": True},
)

print("====================Vector store======================")
m = MongoAtalasStoreService(embeddings)
m.create_collection()
m.create_vector_store()
m.create_retriever()
retriever = m.get_retriever()
print("======================================================\n\n\n")
print("====================Model======================")
q = QwenService()
q.create_model()
res = q.run_chain("Tìm cho tôi luật lao động ở nhật", retriever)
print(res)
print("======================================================\n\n\n")
