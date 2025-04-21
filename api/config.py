from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["http://localhost:3000"]
    
    # Database settings (reuse existing ChromaDB settings)
    CHROMA_DB_DIR: str = "chroma_db"
    
    # API Keys (reuse from .env)
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
