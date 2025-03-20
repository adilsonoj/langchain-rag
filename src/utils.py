import os
from dotenv import load_dotenv

def load_env_vars():
    """Carrega variáveis de ambiente do arquivo .env."""
    load_dotenv()
    return os.getenv("OPENAI_API_KEY"), os.getenv("GROQ_API_KEY"), os.getenv("LANGSMITH_API_KEY")