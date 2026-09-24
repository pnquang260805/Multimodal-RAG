from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from configs.app_config import AppConfig

from services.chunking_service import ChunkingService

conf = AppConfig()

llm = ChatOpenAI(
    model=conf.LLM_MODEL_ID,
    base_url=conf.MODEL_URL,
    api_key=conf.OPEN_AI_API,
    temperature=0.7,
)

embeddings = HuggingFaceEmbeddings(
    model_name=conf.EMB_MODEL_ID,
    # model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

response = llm.invoke("xin chào")
print(type(response))
