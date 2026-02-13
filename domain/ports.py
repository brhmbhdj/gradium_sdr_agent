"""Ports (interfaces) du domaine - Architecture Hexagonale."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .models import Message, Conversation, Prospect, QualificationScore, VoiceConfig


@dataclass
class TranscriptionResult:
    """Résultat d'une transcription STT."""
    text: str
    confidence: float = 0.0
    language: str = "fr"
    duration_seconds: float = 0.0


@dataclass
class SynthesisResult:
    """Résultat d'une synthèse TTS."""
    audio_data: bytes
    audio_url: Optional[str] = None
    duration_seconds: float = 0.0
    format: str = "mp3"


class VoicePort(ABC):
    """Port pour les opérations de reconnaissance et synthèse vocale."""
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str = "fr") -> TranscriptionResult:
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, config: VoiceConfig) -> SynthesisResult:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class LLMPort(ABC):
    """Port pour l'interaction avec les modèles de langage."""
    
    @abstractmethod
    async def generate_response(self, conversation: Conversation, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        pass
    
    @abstractmethod
    async def analyze_intent(self, text: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class QualificationPort(ABC):
    """Port pour la qualification des leads."""
    
    @abstractmethod
    async def qualify(self, conversation: Conversation, prospect: Prospect) -> QualificationScore:
        pass
    
    @abstractmethod
    async def should_transfer(self, conversation: Conversation, score: QualificationScore) -> bool:
        pass


class StoragePort(ABC):
    """Port pour le stockage persistant des données."""
    
    @abstractmethod
    async def save_conversation(self, conversation: Conversation, score: Optional[QualificationScore] = None) -> str:
        pass
    
    @abstractmethod
    async def update_lead_status(self, phone_number: str, status: str, notes: Optional[str] = None) -> bool:
        pass
    
    @abstractmethod
    async def get_lead_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        pass


class TelephonyPort(ABC):
    """Port pour la gestion des appels téléphoniques."""
    
    @abstractmethod
    async def initiate_call(self, to_number: str, from_number: str, webhook_url: str) -> str:
        pass
    
    @abstractmethod
    async def handle_incoming_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def end_call(self, call_id: str) -> bool:
        pass
    
    @abstractmethod
    async def send_sms(self, to_number: str, from_number: str, message: str) -> bool:
        pass
