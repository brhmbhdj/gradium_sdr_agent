"""Configuration de l'application."""

from typing import Optional
from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class QualificationMode(str, Enum):
    MOCK = "mock"
    RESEARCH = "research"


class VoiceProvider(str, Enum):
    GRADIUM = "gradium"
    MOCK = "mock"


class Settings(BaseSettings):
    """Configuration centralisée."""
    
    APP_NAME: str = "Gradium-SDR-Agent"
    APP_ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    GRADIUM_API_KEY: Optional[str] = None
    GRADIUM_API_URL: str = "https://api.gradium.ai/v1"
    VOICE_PROVIDER: VoiceProvider = VoiceProvider.MOCK
    VOICE_SPEED: float = 1.0
    VOICE_LANGUAGE: str = "fr"
    VOICE_ID: str = "nova"
    
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_TEMPERATURE: float = 0.7
    
    SERPAPI_KEY: Optional[str] = None
    QUALIFICATION_MODE: QualificationMode = QualificationMode.MOCK
    
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    TWILIO_WEBHOOK_URL: Optional[str] = None
    
    VAPI_API_KEY: Optional[str] = None
    VAPI_PHONE_NUMBER: Optional[str] = None
    
    NOTION_API_KEY: Optional[str] = None
    NOTION_DATABASE_ID: Optional[str] = None
    NOTION_PARENT_PAGE_ID: Optional[str] = None
    
    NGROK_AUTH_TOKEN: Optional[str] = None
    NGROK_REGION: str = "eu"
    
    WEBHOOK_PORT: int = 8000
    STREAMLIT_PORT: int = 8501
    MAX_CALL_DURATION_MINUTES: int = 5
    QUALIFICATION_THRESHOLD: int = 80
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @property
    def has_gradium(self) -> bool:
        return self.GRADIUM_API_KEY is not None and len(self.GRADIUM_API_KEY) > 0
    
    @property
    def has_gemini(self) -> bool:
        return self.GEMINI_API_KEY is not None and len(self.GEMINI_API_KEY) > 0
    
    @property
    def has_serpapi(self) -> bool:
        return self.SERPAPI_KEY is not None and len(self.SERPAPI_KEY) > 0
    
    @property
    def has_twilio(self) -> bool:
        return all([self.TWILIO_ACCOUNT_SID, self.TWILIO_AUTH_TOKEN, self.TWILIO_PHONE_NUMBER])
    
    @property
    def has_notion(self) -> bool:
        return self.NOTION_API_KEY is not None and len(self.NOTION_API_KEY) > 0
    
    def get_status(self) -> dict:
        return {
            "gradium": self.has_gradium,
            "gemini": self.has_gemini,
            "serpapi": self.has_serpapi,
            "twilio": self.has_twilio,
            "notion": self.has_notion,
            "voice_provider": self.VOICE_PROVIDER.value,
            "qualification_mode": self.QUALIFICATION_MODE.value,
        }


settings = Settings()
