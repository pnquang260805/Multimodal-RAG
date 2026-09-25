from dotenv import load_dotenv
import os

load_dotenv()


class AppConfig:
    def __init__(self):
        self.HUNGGING_FACE_TOKEN = os.getenv("HUNGGING_FACE_TOKEN")
        self.MONGODB_CONNEC_URL = os.getenv("MONGODB_CONNEC_URL")
        self.LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        self.EMB_MODEL_ID = os.getenv("EMB_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
        self.MODEL_URL = os.getenv("MODEL_URL")
        if not self.MODEL_URL.endswith("/v1"):
            self.MODEL_URL += "/v1"
        self.OPEN_AI_API = os.getenv("OPEN_AI_API", "not-needed")
        self.DB_NAME = os.getenv("VECTOR_DB_NAME", "vector_db")
        self.ATLAS_COLLECTION = os.getenv("ATLAS_COLLECTION", "atlas_collection")
