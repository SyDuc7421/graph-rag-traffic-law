from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Chatbot Luật Giao Thông GraphRAG"
    
    # Neo4j Configurations
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # OpenAI Configurations
    OPENAI_API_KEY: str

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
