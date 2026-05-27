from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    pdf_path: Path = Path("Gilbert_Strang_Linear_Algebra_and_Its_Applicatio_230928_225121.pdf")
    index_dir: Path = Path("index")
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gemini-2.5-flash-lite"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 3


def get_settings() -> Settings:
    return Settings()
