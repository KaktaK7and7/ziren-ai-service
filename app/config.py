import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Основная модель для ответов ассистента
    MODEL: str = os.getenv("MODEL", "gpt-4.1-mini")

    # Отдельная модель для анализа долгосрочной памяти
    # Можно оставить такую же, потом заменить на более дешёвую
    MEMORY_MODEL: str = os.getenv("MEMORY_MODEL", "gpt-4.1-mini")

    APP_NAME: str = os.getenv("APP_NAME", "ziren-ai-service")


settings = Settings()