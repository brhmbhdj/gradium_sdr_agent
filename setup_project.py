#!/usr/bin/env python3
"""
Script d'installation automatique du projet Gradium-SDR-Agent.

Ce script:
1. Crée l'arborescence complète des dossiers
2. Génère tous les fichiers Python avec leur contenu
3. Vérifie la présence du fichier .env
4. Teste la connectivité internet

Usage:
    python setup_project.py
"""

import os
import sys
from pathlib import Path


def create_directory_structure(base_path: Path) -> None:
    """Crée la structure de dossiers complète du projet."""
    directories = [
        "domain",
        "application",
        "infrastructure/api",
        "infrastructure/telephony",
        "infrastructure/storage",
        "interface",
        "setup",
        "tests",
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        # Créer __init__.py dans chaque dossier
        init_file = dir_path / "__init__.py"
        init_file.touch(exist_ok=True)
    
    print("✅ Structure de dossiers créée")


def write_file(base_path: Path, relative_path: str, content: str) -> None:
    """Écrit un fichier avec son contenu."""
    file_path = base_path / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📝 Créé: {relative_path}")


def main():
    """Point d'entrée principal."""
    print("="*70)
    print("🚀 GRADIUM-SDR-AGENT - Script d'installation")
    print("="*70)
    print()
    
    base_path = Path(__file__).parent
    print(f"📁 Répertoire de destination: {base_path.absolute()}")
    print()
    
    # Créer la structure
    create_directory_structure(base_path)
    
    # Créer tous les fichiers
    create_domain_files(base_path)
    create_application_files(base_path)
    create_infrastructure_files(base_path)
    create_interface_files(base_path)
    create_setup_files(base_path)
    create_test_files(base_path)
    create_config_files(base_path)
    create_readme(base_path)
    
    print()
    print("="*70)
    print("✅ INSTALLATION TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print()
    print("📋 Prochaines étapes:")
    print("   1. cd gradium_sdr_agent")
    print("   2. python -m venv venv")
    print("   3. source venv/bin/activate")
    print("   4. pip install -r requirements.txt")
    print("   5. cp .env.example .env")
    print("   6. Éditez .env avec vos clés API")
    print("   7. python setup/test_setup.py")
    print()
    print("📖 Consultez le README.md pour plus de détails")
    print()


def create_domain_files(base_path: Path) -> None:
    """Crée les fichiers du domaine."""
    
    # domain/models.py
    models_content = '''"""Modèles de domaine pour Gradium-SDR-Agent."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class Language(Enum):
    """Langues supportées pour la synthèse vocale."""
    FR = "fr"
    EN = "en"
    ES = "es"
    DE = "de"
    PT = "pt"


class VoiceStyle(Enum):
    """Styles de voix pour l'interaction."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ENERGETIC = "energetic"


class LeadStatus(Enum):
    """Statuts possibles d'un lead dans le pipeline."""
    NOUVEAU = "Nouveau"
    EN_COURS = "En cours"
    QUALIFIE = "Qualifié"
    TRANSFERE = "Transféré"
    NON_QUALIFIE = "Non qualifié"


@dataclass
class VoiceConfig:
    """Configuration de la voix pour les appels."""
    speed: float = 1.0
    language: Language = Language.FR
    voice_id: str = "nova"
    gender: str = "female"
    style: VoiceStyle = VoiceStyle.PROFESSIONAL
    pitch: float = 0.0
    
    def __post_init__(self):
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("speed doit être entre 0.5 et 2.0")
        if self.voice_id not in ["luna", "nova", "echo", "onyx"]:
            raise ValueError("voice_id doit être luna, nova, echo ou onyx")


@dataclass
class Message:
    """Message dans une conversation."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    audio_url: Optional[str] = None
    
    def __post_init__(self):
        if self.role not in ["user", "assistant", "system"]:
            raise ValueError("role doit être user, assistant ou system")


@dataclass
class Conversation:
    """Conversation complète avec un prospect."""
    id: str
    phone_number: str
    messages: List[Message] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, audio_url: Optional[str] = None) -> None:
        self.messages.append(Message(role=role, content=content, audio_url=audio_url))
    
    def get_transcript(self) -> str:
        lines = []
        for msg in self.messages:
            speaker = "Prospect" if msg.role == "user" else "Agent"
            lines.append(f"{speaker}: {msg.content}")
        return "\\n".join(lines)
    
    def end_conversation(self, status: str = "completed") -> None:
        self.ended_at = datetime.now()
        self.status = status


@dataclass
class Prospect:
    """Informations sur un prospect."""
    phone_number: str
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    source: str = "inconnu"
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualificationScore:
    """Score de qualification d'un lead."""
    total_score: int = 0
    budget_score: int = 0
    authority_score: int = 0
    need_score: int = 0
    timeline_score: int = 0
    is_qualified: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total(self) -> int:
        self.total_score = self.budget_score + self.authority_score + self.need_score + self.timeline_score
        self.is_qualified = self.total_score >= 80
        return self.total_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "budget_score": self.budget_score,
            "authority_score": self.authority_score,
            "need_score": self.need_score,
            "timeline_score": self.timeline_score,
            "is_qualified": self.is_qualified,
            "details": self.details,
        }
'''
    write_file(base_path, "domain/models.py", models_content)
    
    # domain/ports.py
    ports_content = '''"""Ports (interfaces) du domaine - Architecture Hexagonale."""

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
'''
    write_file(base_path, "domain/ports.py", ports_content)
    
    # domain/qualification.py
    qualification_content = '''"""Logique métier de qualification des leads."""

from typing import Dict, List, Any, Optional

from .models import Conversation, Prospect, QualificationScore
from .ports import QualificationPort


class RuleBasedQualifier(QualificationPort):
    """Qualificateur basé sur des règles simples."""
    
    KEYWORDS = {
        "budget": ["budget", "euros", "€", "dollars", "$", "investir", "investissement", "prix", "coût", "tarif", "combien"],
        "authority": ["décide", "décision", "décideur", "responsable", "directeur", "manager", "chef", "ceo", "fondateur"],
        "need": ["besoin", "problème", "solution", "cherchons", "intéressé", "améliorer", "optimiser"],
        "timeline": ["délai", "timeline", "quand", "date", "mois", "semaine", "urgent", "rapidement"],
        "transfer": ["transfert", "transférer", "parler à", "humain", "conseiller", "commercial"]
    }
    
    def __init__(self, qualification_threshold: int = 80):
        self.qualification_threshold = qualification_threshold
    
    def _analyze_category(self, conversation: Conversation, category: str) -> tuple:
        keywords = self.KEYWORDS.get(category, [])
        found_keywords = []
        transcript = conversation.get_transcript().lower()
        
        for keyword in keywords:
            if keyword.lower() in transcript:
                found_keywords.append(keyword)
        
        score = min(len(found_keywords) * 5, 25)
        return score, found_keywords
    
    async def qualify(self, conversation: Conversation, prospect: Prospect) -> QualificationScore:
        score = QualificationScore()
        details = {}
        
        score.budget_score, details["budget_keywords"] = self._analyze_category(conversation, "budget")
        score.authority_score, details["authority_keywords"] = self._analyze_category(conversation, "authority")
        score.need_score, details["need_keywords"] = self._analyze_category(conversation, "need")
        score.timeline_score, details["timeline_keywords"] = self._analyze_category(conversation, "timeline")
        
        score.calculate_total()
        score.details = details
        
        return score
    
    async def should_transfer(self, conversation: Conversation, score: QualificationScore) -> bool:
        if score.total_score >= self.qualification_threshold:
            return True
        
        transcript = conversation.get_transcript().lower()
        transfer_keywords = self.KEYWORDS.get("transfer", [])
        
        for keyword in transfer_keywords:
            if keyword.lower() in transcript:
                return True
        
        return False


class ResearchBasedQualifier(RuleBasedQualifier):
    """Qualificateur enrichi par la recherche web."""
    
    def __init__(self, qualification_threshold: int = 80, serpapi_client: Optional[Any] = None):
        super().__init__(qualification_threshold)
        self.serpapi_client = serpapi_client
    
    async def research_company(self, company_name: str) -> Dict[str, Any]:
        if not self.serpapi_client:
            return {"error": "SerpAPI client not configured"}
        
        try:
            results = await self.serpapi_client.search(query=f"{company_name} entreprise", num_results=5)
            return {"company": company_name, "results": results}
        except Exception as e:
            return {"error": str(e), "company": company_name}
    
    async def qualify(self, conversation: Conversation, prospect: Prospect) -> QualificationScore:
        score = await super().qualify(conversation, prospect)
        
        if prospect.company and self.serpapi_client:
            research_data = await self.research_company(prospect.company)
            score.details["research"] = research_data
            
            if "results" in research_data and len(research_data["results"]) > 0:
                score.authority_score = min(score.authority_score + 5, 25)
                score.details["company_verified"] = True
        
        score.calculate_total()
        return score
'''
    write_file(base_path, "domain/qualification.py", qualification_content)
    
    print("✅ Fichiers du domaine créés")


def create_application_files(base_path: Path) -> None:
    """Crée les fichiers de la couche application."""
    
    service_content = '''"""Service d'orchestration des conversations."""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from domain.models import Conversation, Prospect, VoiceConfig, LeadStatus
from domain.ports import VoicePort, LLMPort, QualificationPort, StoragePort, TelephonyPort

logger = logging.getLogger(__name__)


class ConversationService:
    """Service principal d'orchestration des conversations."""
    
    def __init__(
        self,
        voice_port: VoicePort,
        llm_port: LLMPort,
        qualification_port: QualificationPort,
        storage_port: StoragePort,
        telephony_port: Optional[TelephonyPort] = None,
        voice_config: Optional[VoiceConfig] = None
    ):
        self.voice_port = voice_port
        self.llm_port = llm_port
        self.qualification_port = qualification_port
        self.storage_port = storage_port
        self.telephony_port = telephony_port
        self.voice_config = voice_config or VoiceConfig()
        self._active_conversations: Dict[str, Conversation] = {}
        logger.info("ConversationService initialisé")
    
    async def start_conversation(self, phone_number: str, prospect_info: Optional[Dict[str, Any]] = None) -> Conversation:
        conversation_id = str(uuid.uuid4())
        
        conversation = Conversation(
            id=conversation_id,
            phone_number=phone_number,
            metadata={"prospect_info": prospect_info or {}, "started_at": datetime.now().isoformat()}
        )
        
        self._active_conversations[conversation_id] = conversation
        await self.storage_port.save_conversation(conversation)
        
        logger.info(f"Conversation démarrée: {conversation_id} pour {phone_number}")
        return conversation
    
    async def process_audio_input(self, conversation_id: str, audio_data: bytes, language: str = "fr"):
        conversation = self._get_conversation(conversation_id)
        
        try:
            transcription = await self.voice_port.transcribe(audio_data, language)
            conversation.add_message("user", transcription.text)
            
            response_text = await self.llm_port.generate_response(conversation)
            conversation.add_message("assistant", response_text)
            
            synthesis = await self.voice_port.synthesize(response_text, self.voice_config)
            await self.storage_port.save_conversation(conversation)
            
            return synthesis
        except Exception as e:
            logger.error(f"Erreur traitement audio: {e}")
            error_message = "Je suis désolé, je n'ai pas compris. Pouvez-vous répéter ?"
            return await self.voice_port.synthesize(error_message, self.voice_config)
    
    async def process_text_input(self, conversation_id: str, text: str) -> str:
        conversation = self._get_conversation(conversation_id)
        conversation.add_message("user", text)
        
        response_text = await self.llm_port.generate_response(conversation)
        conversation.add_message("assistant", response_text)
        
        await self.storage_port.save_conversation(conversation)
        return response_text
    
    async def qualify_lead(self, conversation_id: str, prospect: Optional[Prospect] = None) -> Dict[str, Any]:
        conversation = self._get_conversation(conversation_id)
        
        if prospect is None:
            prospect = Prospect(phone_number=conversation.phone_number)
        
        score = await self.qualification_port.qualify(conversation, prospect)
        should_transfer = await self.qualification_port.should_transfer(conversation, score)
        
        status = LeadStatus.QUALIFIE.value if score.is_qualified else LeadStatus.EN_COURS.value
        await self.storage_port.update_lead_status(conversation.phone_number, status, notes=f"Score: {score.total_score}/100")
        await self.storage_port.save_conversation(conversation, score)
        
        return {
            "conversation_id": conversation_id,
            "score": score.to_dict(),
            "should_transfer": should_transfer,
            "qualified": score.is_qualified
        }
    
    async def end_conversation(self, conversation_id: str, reason: str = "completed") -> Dict[str, Any]:
        conversation = self._get_conversation(conversation_id)
        conversation.end_conversation(status=reason)
        
        prospect = Prospect(phone_number=conversation.phone_number)
        score = await self.qualification_port.qualify(conversation, prospect)
        await self.storage_port.save_conversation(conversation, score)
        
        del self._active_conversations[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "phone_number": conversation.phone_number,
            "duration_seconds": (conversation.ended_at - conversation.started_at).total_seconds() if conversation.ended_at else 0,
            "message_count": len(conversation.messages),
            "final_score": score.to_dict(),
            "transcript": conversation.get_transcript(),
            "reason": reason
        }
    
    def _get_conversation(self, conversation_id: str) -> Conversation:
        if conversation_id not in self._active_conversations:
            raise ValueError(f"Conversation non trouvée: {conversation_id}")
        return self._active_conversations[conversation_id]
    
    def get_active_conversations(self) -> Dict[str, Conversation]:
        return self._active_conversations.copy()
'''
    write_file(base_path, "application/conversation_service.py", service_content)
    
    print("✅ Fichiers de l'application créés")


def create_infrastructure_files(base_path: Path) -> None:
    """Crée les fichiers de la couche infrastructure."""
    
    # infrastructure/config.py
    config_content = '''"""Configuration de l'application."""

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
'''
    write_file(base_path, "infrastructure/config.py", config_content)
    
    # infrastructure/api/gradium_adapter.py
    gradium_content = '''"""Adaptateur Gradium - Implémentation du VoicePort."""

import logging
from typing import Optional
from pathlib import Path

import httpx

from domain.models import VoiceConfig
from domain.ports import VoicePort, TranscriptionResult, SynthesisResult
from infrastructure.config import settings

logger = logging.getLogger(__name__)


class GradiumAdapter(VoicePort):
    """Adaptateur pour l'API Gradium (STT + TTS)."""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or settings.GRADIUM_API_KEY
        self.api_url = api_url or settings.GRADIUM_API_URL
        self.mock_mode = not self.api_key
        self.audio_dir = Path("audio")
        
        if self.mock_mode:
            logger.warning("Mode MOCK activé pour Gradium")
            self.audio_dir.mkdir(exist_ok=True)
        else:
            logger.info("GradiumAdapter initialisé avec API key")
    
    def is_available(self) -> bool:
        return True
    
    async def transcribe(self, audio_data: bytes, language: str = "fr") -> TranscriptionResult:
        if self.mock_mode:
            return await self._mock_transcribe(audio_data, language)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/speech-to-text",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"audio": ("audio.wav", audio_data, "audio/wav")},
                    data={"language": language},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return TranscriptionResult(
                    text=result.get("text", ""),
                    confidence=result.get("confidence", 0.9),
                    language=language,
                    duration_seconds=result.get("duration", 0.0)
                )
        except Exception as e:
            logger.error(f"Erreur transcription Gradium: {e}")
            return await self._mock_transcribe(audio_data, language)
    
    async def _mock_transcribe(self, audio_data: bytes, language: str) -> TranscriptionResult:
        demo_messages = {
            "fr": "Bonjour, je suis intéressé par votre solution. Quel est le prix ?",
            "en": "Hello, I'm interested in your solution. What's the price?",
        }
        text = demo_messages.get(language, demo_messages["fr"])
        
        return TranscriptionResult(
            text=text,
            confidence=0.85,
            language=language,
            duration_seconds=len(audio_data) / 16000
        )
    
    async def synthesize(self, text: str, config: VoiceConfig) -> SynthesisResult:
        if self.mock_mode:
            return await self._mock_synthesize(text, config)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/text-to-speech",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"text": text, "voice": config.voice_id, "speed": config.speed, "language": config.language.value},
                    timeout=30.0
                )
                response.raise_for_status()
                audio_data = response.content
                
                return SynthesisResult(audio_data=audio_data, duration_seconds=len(text) * 0.08, format="mp3")
        except Exception as e:
            logger.error(f"Erreur synthèse Gradium: {e}")
            return await self._mock_synthesize(text, config)
    
    async def _mock_synthesize(self, text: str, config: VoiceConfig) -> SynthesisResult:
        mock_audio = b"\\xff\\xf3\\x44\\xc0" + b"\\x00" * 417
        return SynthesisResult(audio_data=mock_audio, duration_seconds=len(text) * 0.08, format="mp3")
'''
    write_file(base_path, "infrastructure/api/gradium_adapter.py", gradium_content)
    
    # infrastructure/api/gemini_adapter.py
    gemini_content = '''"""Adaptateur Google Gemini - Implémentation du LLMPort."""

import logging
from typing import Optional, Dict, Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from domain.models import Conversation
from domain.ports import LLMPort
from infrastructure.config import settings

logger = logging.getLogger(__name__)


class GeminiAdapter(LLMPort):
    """Adaptateur pour l'API Google Gemini."""
    
    DEFAULT_SYSTEM_PROMPT = """Tu es un SDR professionnel pour une entreprise B2B.

Ton rôle:
1. Accueillir chaleureusement le prospect
2. Comprendre son besoin principal (BANT)
3. Poser des questions ouvertes pour qualifier
4. Proposer une démo si le lead est chaud

Règles:
- Sois concis (max 2-3 phrases)
- Sois professionnel mais chaleureux
- Ne ment jamais
- Réponds en français"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model or settings.GEMINI_MODEL
        self.temperature = temperature or settings.GEMINI_TEMPERATURE
        self.mock_mode = not self.api_key
        
        if not self.mock_mode:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"GeminiAdapter initialisé avec modèle {self.model_name}")
            except Exception as e:
                logger.error(f"Erreur initialisation Gemini: {e}")
                self.mock_mode = True
        else:
            logger.warning("Mode MOCK activé pour Gemini")
    
    def is_available(self) -> bool:
        return True
    
    async def generate_response(self, conversation: Conversation, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        if self.mock_mode:
            return self._mock_response(conversation)
        
        try:
            messages = self._build_messages(conversation, system_prompt)
            generation_config = GenerationConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=256,
                top_p=0.9,
                top_k=40
            )
            
            response = self.model.generate_content(messages, generation_config=generation_config)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Erreur génération Gemini: {e}")
            return self._mock_response(conversation)
    
    def _build_messages(self, conversation: Conversation, system_prompt: Optional[str]):
        prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        messages = [f"Instructions: {prompt}"]
        
        for msg in conversation.messages:
            role_prefix = "Prospect" if msg.role == "user" else "SDR"
            messages.append(f"{role_prefix}: {msg.content}")
        
        messages.append("SDR:")
        return "\\n".join(messages)
    
    def _mock_response(self, conversation: Conversation) -> str:
        message_count = len(conversation.messages)
        
        if message_count <= 1:
            return "Bonjour ! Merci pour votre intérêt. Je vois que vous avez téléchargé notre contenu. Pouvez-vous me dire ce qui vous intéresse ?"
        elif message_count <= 3:
            return "Je comprends. Pour mieux vous aider, pourriez-vous me dire quel est votre rôle dans l'entreprise ?"
        elif message_count <= 5:
            return "Excellent ! Et concernant votre budget pour ce type de projet, avez-vous une enveloppe définie ?"
        else:
            return "Parfait, je vais vous transférer à l'un de nos experts pour une démonstration personnalisée."
    
    async def analyze_intent(self, text: str) -> Dict[str, Any]:
        if self.mock_mode:
            return self._mock_analyze_intent(text)
        
        try:
            prompt = f"Analyse l'intention: '{text}'. Réponds en JSON avec intent, confidence, entities."
            response = self.model.generate_content(prompt)
            import json
            return json.loads(response.text)
        except Exception as e:
            return self._mock_analyze_intent(text)
    
    def _mock_analyze_intent(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        if any(word in text_lower for word in ["prix", "coût", "budget", "€"]):
            return {"intent": "information", "confidence": 0.8, "entities": {}}
        elif any(word in text_lower for word in ["transfert", "humain", "commercial"]):
            return {"intent": "transfer", "confidence": 0.85, "entities": {}}
        return {"intent": "information", "confidence": 0.6, "entities": {}}
'''
    write_file(base_path, "infrastructure/api/gemini_adapter.py", gemini_content)
    
    # infrastructure/api/serpapi_client.py
    serpapi_content = '''"""Client SerpAPI - Recherche web pour qualification avancée."""

import logging
from typing import Dict, Any, List, Optional

import httpx

from infrastructure.config import settings

logger = logging.getLogger(__name__)


class SerpAPIClient:
    """Client pour l'API SerpAPI."""
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SERPAPI_KEY
        self.enabled = self.api_key is not None
        
        if self.enabled:
            logger.info("SerpAPIClient initialisé")
        else:
            logger.warning("SerpAPI désactivé")
    
    async def search(self, query: str, num_results: int = 5, location: str = "France") -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        
        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": num_results,
                "location": location,
                "hl": "fr",
                "gl": "fr"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(self.BASE_URL, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                organic_results = data.get("organic_results", [])
                formatted_results = []
                for result in organic_results[:num_results]:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    })
                
                return formatted_results
        except Exception as e:
            logger.error(f"Erreur recherche SerpAPI: {e}")
            return []
    
    async def search_company(self, company_name: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"error": "SerpAPI not configured"}
        
        results = await self.search(f"{company_name} entreprise", num_results=5)
        return {"name": company_name, "results": results}
'''
    write_file(base_path, "infrastructure/api/serpapi_client.py", serpapi_content)
    
    # infrastructure/telephony/twilio_adapter.py
    twilio_content = '''"""Adaptateur Twilio - Implémentation du TelephonyPort."""

import logging
from typing import Dict, Any, Optional

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Say, Gather

from domain.ports import TelephonyPort
from infrastructure.config import settings

logger = logging.getLogger(__name__)


class TwilioAdapter(TelephonyPort):
    """Adaptateur pour l'API Twilio."""
    
    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None, from_number: Optional[str] = None):
        self.account_sid = account_sid or settings.TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or settings.TWILIO_AUTH_TOKEN
        self.from_number = from_number or settings.TWILIO_PHONE_NUMBER
        
        self.enabled = all([self.account_sid, self.auth_token, self.from_number])
        
        if self.enabled:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info(f"TwilioAdapter initialisé")
            except Exception as e:
                logger.error(f"Erreur initialisation Twilio: {e}")
                self.enabled = False
                self.client = None
        else:
            logger.warning("Twilio désactivé")
            self.client = None
    
    def is_available(self) -> bool:
        return self.enabled and self.client is not None
    
    async def initiate_call(self, to_number: str, from_number: str, webhook_url: str) -> str:
        if not self.is_available():
            raise RuntimeError("Twilio n'est pas configuré")
        
        try:
            call = self.client.calls.create(
                to=to_number,
                from_=from_number or self.from_number,
                url=webhook_url,
                method="POST"
            )
            logger.info(f"Appel Twilio initié: {call.sid}")
            return call.sid
        except Exception as e:
            logger.error(f"Erreur appel Twilio: {e}")
            raise
    
    async def handle_incoming_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        call_sid = payload.get("CallSid", "unknown")
        logger.info(f"Webhook Twilio reçu: {call_sid}")
        
        response = VoiceResponse()
        response.say("Bonjour, je suis votre assistant virtuel.", language="fr-FR", voice="Polly.Celine")
        
        gather = Gather(input="speech", language="fr-FR", speech_timeout="auto", action="/webhook/twilio/gather", method="POST")
        gather.say("Comment puis-je vous aider ?", language="fr-FR")
        response.append(gather)
        
        return {"twiml": str(response), "content_type": "application/xml"}
    
    async def end_call(self, call_id: str) -> bool:
        if not self.is_available():
            return False
        try:
            self.client.calls(call_id).update(status="completed")
            return True
        except Exception as e:
            logger.error(f"Erreur terminaison: {e}")
            return False
    
    async def send_sms(self, to_number: str, from_number: str, message: str) -> bool:
        if not self.is_available():
            return False
        try:
            self.client.messages.create(to=to_number, from_=from_number or self.from_number, body=message)
            return True
        except Exception as e:
            logger.error(f"Erreur SMS: {e}")
            return False
'''
    write_file(base_path, "infrastructure/telephony/twilio_adapter.py", twilio_content)
    
    # infrastructure/telephony/vapi_adapter.py
    vapi_content = '''"""Adaptateur VAPI - Alternative à Twilio."""

import logging
from typing import Dict, Any, Optional

import httpx

from domain.ports import TelephonyPort
from infrastructure.config import settings

logger = logging.getLogger(__name__)


class VapiAdapter(TelephonyPort):
    """Adaptateur pour l'API VAPI."""
    
    BASE_URL = "https://api.vapi.ai"
    
    def __init__(self, api_key: Optional[str] = None, phone_number: Optional[str] = None):
        self.api_key = api_key or settings.VAPI_API_KEY
        self.phone_number = phone_number or settings.VAPI_PHONE_NUMBER
        self.enabled = self.api_key is not None
        
        if self.enabled:
            logger.info("VapiAdapter initialisé")
        else:
            logger.warning("VAPI désactivé")
    
    def is_available(self) -> bool:
        return self.enabled
    
    async def initiate_call(self, to_number: str, from_number: str, webhook_url: str) -> str:
        if not self.is_available():
            raise RuntimeError("VAPI n'est pas configuré")
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "assistant": {"firstMessage": "Bonjour, comment puis-je vous aider ?"},
            "phoneNumber": {"twilioPhoneNumber": from_number or self.phone_number},
            "customer": {"number": to_number},
            "webhookUrl": webhook_url
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.BASE_URL}/call", headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            return result.get("id", "unknown")
    
    async def handle_incoming_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "received"}
    
    async def end_call(self, call_id: str) -> bool:
        if not self.is_available():
            return False
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{self.BASE_URL}/call/{call_id}", headers=headers, timeout=10.0)
            return response.status_code == 200
    
    async def send_sms(self, to_number: str, from_number: str, message: str) -> bool:
        logger.warning("SMS non supporté par VAPI")
        return False
'''
    write_file(base_path, "infrastructure/telephony/vapi_adapter.py", vapi_content)
    
    # infrastructure/storage/notion_adapter.py
    notion_content = '''"""Adaptateur Notion - Implémentation du StoragePort."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

import httpx

from domain.models import Conversation, QualificationScore, LeadStatus
from domain.ports import StoragePort
from infrastructure.config import settings

logger = logging.getLogger(__name__)


class NotionAdapter(StoragePort):
    """Adaptateur pour le stockage dans Notion."""
    
    BASE_URL = "https://api.notion.com/v1"
    API_VERSION = "2022-06-28"
    
    def __init__(self, api_key: Optional[str] = None, database_id: Optional[str] = None):
        self.api_key = api_key or settings.NOTION_API_KEY
        self.database_id = database_id or settings.NOTION_DATABASE_ID
        self.enabled = self.api_key is not None
        
        if self.enabled:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Notion-Version": self.API_VERSION
            }
            logger.info("NotionAdapter initialisé")
        else:
            logger.warning("Notion désactivé")
    
    async def save_conversation(self, conversation: Conversation, score: Optional[QualificationScore] = None) -> str:
        if not self.enabled or not self.database_id:
            logger.warning("Notion non configuré")
            return "mock_id"
        
        try:
            properties = self._build_page_properties(conversation, score)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/pages",
                    headers=self.headers,
                    json={"parent": {"database_id": self.database_id}, "properties": properties},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result.get("id", "unknown")
        except Exception as e:
            logger.error(f"Erreur sauvegarde Notion: {e}")
            return "error"
    
    def _build_page_properties(self, conversation: Conversation, score: Optional[QualificationScore]):
        status = LeadStatus.QUALIFIE.value if score and score.is_qualified else LeadStatus.EN_COURS.value
        prospect_info = conversation.metadata.get("prospect_info", {})
        prospect_name = prospect_info.get("name", f"Lead {conversation.phone_number}")
        
        return {
            "Nom": {"title": [{"text": {"content": prospect_name}}]},
            "Téléphone": {"phone_number": conversation.phone_number},
            "Score": {"number": score.total_score if score else 0},
            "Statut": {"select": {"name": status}},
            "Transcript": {"rich_text": [{"text": {"content": conversation.get_transcript()[:2000]}}]},
            "Date": {"date": {"start": conversation.started_at.isoformat()}}
        }
    
    async def update_lead_status(self, phone_number: str, status: str, notes: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        
        try:
            page_id = await self._find_page_by_phone(phone_number)
            if not page_id:
                return False
            
            update_data = {"properties": {"Statut": {"select": {"name": status}}}}
            if notes:
                update_data["properties"]["Notes"] = {"rich_text": [{"text": {"content": notes}}]}
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.BASE_URL}/pages/{page_id}",
                    headers=self.headers,
                    json=update_data,
                    timeout=30.0
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Erreur mise à jour Notion: {e}")
            return False
    
    async def _find_page_by_phone(self, phone_number: str) -> Optional[str]:
        if not self.database_id:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/databases/{self.database_id}/query",
                    headers=self.headers,
                    json={"filter": {"property": "Téléphone", "phone_number": {"equals": phone_number}}},
                    timeout=30.0
                )
                result = response.json()
                results = result.get("results", [])
                return results[0].get("id") if results else None
        except Exception as e:
            logger.error(f"Erreur recherche Notion: {e}")
            return None
    
    async def get_lead_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        
        page_id = await self._find_page_by_phone(phone_number)
        if not page_id:
            return None
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/pages/{page_id}", headers=self.headers, timeout=30.0)
            return response.json()
'''
    write_file(base_path, "infrastructure/storage/notion_adapter.py", notion_content)
    
    # infrastructure/storage/salesforce_mock.py
    salesforce_content = '''"""Mock Salesforce - Implémentation mock du StoragePort."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from domain.models import Conversation, QualificationScore
from domain.ports import StoragePort

logger = logging.getLogger(__name__)


class SalesforceMock(StoragePort):
    """Mock de stockage Salesforce."""
    
    def __init__(self):
        self.storage: Dict[str, Dict[str, Any]] = {}
        logger.info("SalesforceMock initialisé")
    
    async def save_conversation(self, conversation: Conversation, score: Optional[QualificationScore] = None) -> str:
        record_id = f"MOCK_{conversation.id}"
        self.storage[record_id] = {
            "id": record_id,
            "phone_number": conversation.phone_number,
            "transcript": conversation.get_transcript(),
            "message_count": len(conversation.messages),
            "score": score.to_dict() if score else None,
            "saved_at": datetime.now().isoformat()
        }
        logger.info(f"[MOCK] Conversation sauvegardée: {record_id}")
        return record_id
    
    async def update_lead_status(self, phone_number: str, status: str, notes: Optional[str] = None) -> bool:
        logger.info(f"[MOCK] Statut mis à jour pour {phone_number}: {status}")
        return True
    
    async def get_lead_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        for record_id, data in self.storage.items():
            if data.get("phone_number") == phone_number:
                return data
        return None
'''
    write_file(base_path, "infrastructure/storage/salesforce_mock.py", salesforce_content)
    
    print("✅ Fichiers infrastructure créés")


def create_interface_files(base_path: Path) -> None:
    """Crée les fichiers de l'interface."""
    
    # interface/streamlit_dashboard.py
    streamlit_content = '''"""Dashboard Streamlit - Interface de monitoring."""

import logging
from datetime import datetime

import streamlit as st
import pandas as pd

import sys
sys.path.insert(0, ".")

from infrastructure.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Gradium SDR Agent", page_icon="🎙️", layout="wide")


def main():
    st.title("🎙️ Gradium SDR Agent")
    st.markdown("### Dashboard de monitoring temps réel")
    st.divider()
    
    # Statut des services
    st.subheader("🔌 Statut des Services")
    status = settings.get_status()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🗣️ Voix**")
        st.success("✅ Gradium") if status["gradium"] else st.warning("⚠️ Mode Mock")
    with col2:
        st.markdown("**🤖 LLM**")
        st.success("✅ Gemini") if status["gemini"] else st.warning("⚠️ Mode Mock")
    with col3:
        st.markdown("**📞 Téléphonie**")
        st.success("✅ Configuré") if status["twilio"] else st.error("❌ Non configuré")
    
    st.divider()
    
    # Statistiques
    st.subheader("📈 Statistiques")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Appels aujourd'hui", "12", "+3")
    c2.metric("Leads qualifiés", "5", "+2")
    c3.metric("Taux de qualification", "42%", "+5%")
    c4.metric("Score moyen", "68", "+3")
    
    st.divider()
    
    # Conversations actives
    st.subheader("📞 Conversations Actives")
    demo_data = [
        {"ID": "conv_001", "Téléphone": "+33 6 12 34 56 78", "Durée": "2:34", "Messages": 5, "Score": 65, "Statut": "En cours"},
        {"ID": "conv_002", "Téléphone": "+33 7 98 76 54 32", "Durée": "1:12", "Messages": 3, "Score": 45, "Statut": "En cours"}
    ]
    df = pd.DataFrame(demo_data)
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    # Leads qualifiés
    st.subheader("⭐ Leads Qualifiés")
    demo_leads = [
        {"Nom": "Jean Dupont", "Entreprise": "TechCorp", "Téléphone": "+33 6 11 22 33 44", "Score": 85, "Statut": "Transféré"},
        {"Nom": "Marie Martin", "Entreprise": "StartupXYZ", "Téléphone": "+33 7 55 66 77 88", "Score": 92, "Statut": "Qualifié"}
    ]
    df_leads = pd.DataFrame(demo_leads)
    st.dataframe(df_leads, use_container_width=True)
    
    st.divider()
    st.caption("Gradium-SDR-Agent v1.0")


if __name__ == "__main__":
    main()
'''
    write_file(base_path, "interface/streamlit_dashboard.py", streamlit_content)
    
    # interface/webhook_server.py
    webhook_content = '''"""Serveur Webhook FastAPI."""

import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, ".")

from infrastructure.config import settings
from infrastructure.api.gradium_adapter import GradiumAdapter
from infrastructure.api.gemini_adapter import GeminiAdapter
from infrastructure.storage.salesforce_mock import SalesforceMock
from infrastructure.storage.notion_adapter import NotionAdapter
from domain.qualification import RuleBasedQualifier, ResearchBasedQualifier
from domain.models import VoiceConfig
from application.conversation_service import ConversationService

conversation_service: Optional[ConversationService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global conversation_service
    logger.info("🚀 Démarrage du serveur webhook...")
    
    voice_port = GradiumAdapter()
    llm_port = GeminiAdapter()
    storage_port = NotionAdapter() if settings.has_notion else SalesforceMock()
    
    if settings.QUALIFICATION_MODE.value == "research" and settings.has_serpapi:
        from infrastructure.api.serpapi_client import SerpAPIClient
        qualification_port = ResearchBasedQualifier(serpapi_client=SerpAPIClient())
    else:
        qualification_port = RuleBasedQualifier()
    
    voice_config = VoiceConfig(speed=settings.VOICE_SPEED, language=settings.VOICE_LANGUAGE, voice_id=settings.VOICE_ID)
    
    conversation_service = ConversationService(
        voice_port=voice_port,
        llm_port=llm_port,
        qualification_port=qualification_port,
        storage_port=storage_port,
        voice_config=voice_config
    )
    
    logger.info("✅ Services initialisés")
    yield
    logger.info("🛑 Arrêt du serveur")


app = FastAPI(title="Gradium SDR Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"message": "Gradium SDR Agent", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": settings.get_status()}


@app.get("/status")
async def get_status():
    return {"app_name": settings.APP_NAME, "environment": settings.APP_ENV, "services": settings.get_status()}


@app.post("/webhook/twilio", response_class=PlainTextResponse)
async def twilio_webhook(CallSid: str = Form(...), From: str = Form(...), To: str = Form(...), CallStatus: str = Form(default="unknown")):
    logger.info(f"📞 Twilio webhook: {CallSid} from {From}")
    
    conv = await conversation_service.start_conversation(phone_number=From, prospect_info={"source": "twilio_inbound"})
    
    from twilio.twiml.voice_response import VoiceResponse, Gather
    response = VoiceResponse()
    response.say("Bonjour ! Je suis votre assistant virtuel. Comment puis-je vous aider ?", language="fr-FR", voice="Polly.Celine")
    
    gather = Gather(input="speech", language="fr-FR", speech_timeout="auto", action=f"/webhook/twilio/gather?conv_id={conv.id}", method="POST")
    response.append(gather)
    
    return str(response)


@app.post("/webhook/twilio/gather", response_class=PlainTextResponse)
async def twilio_gather(request: Request, conv_id: str, SpeechResult: str = Form(default="")):
    logger.info(f"🎤 Speech: '{SpeechResult}'")
    
    from twilio.twiml.voice_response import VoiceResponse, Gather
    
    if not SpeechResult:
        response = VoiceResponse()
        response.say("Je n'ai pas entendu. Pouvez-vous répéter ?", language="fr-FR")
        gather = Gather(input="speech", language="fr-FR", speech_timeout="auto", action=f"/webhook/twilio/gather?conv_id={conv_id}", method="POST")
        response.append(gather)
        return str(response)
    
    response_text = await conversation_service.process_text_input(conversation_id=conv_id, text=SpeechResult)
    
    response = VoiceResponse()
    response.say(response_text, language="fr-FR", voice="Polly.Celine")
    
    gather = Gather(input="speech", language="fr-FR", speech_timeout="auto", action=f"/webhook/twilio/gather?conv_id={conv_id}", method="POST")
    response.append(gather)
    
    return str(response)


@app.post("/webhook/vapi")
async def vapi_webhook(request: Request):
    payload = await request.json()
    logger.info(f"📞 VAPI webhook: {payload.get('type', 'unknown')}")
    return {"status": "received"}


@app.get("/api/conversations")
async def get_conversations():
    if conversation_service:
        conversations = conversation_service.get_active_conversations()
        return {"count": len(conversations), "conversations": [{"id": c.id, "phone": c.phone_number} for c in conversations.values()]}
    return {"count": 0, "conversations": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=settings.WEBHOOK_PORT, reload=True)
'''
    write_file(base_path, "interface/webhook_server.py", webhook_content)
    
    print("✅ Fichiers interface créés")


def create_setup_files(base_path: Path) -> None:
    """Crée les fichiers setup."""
    
    # setup/create_notion_db.py
    notion_db_content = '''"""Script de création de la database Notion."""

import os
import sys
import argparse
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NOTION_API_KEY")
PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID")
BASE_URL = "https://api.notion.com/v1"
API_VERSION = "2022-06-28"


def create_voice_leads_database(api_key: str, parent_page_id: Optional[str] = None) -> str:
    if not parent_page_id:
        print("⚠️  NOTION_PARENT_PAGE_ID non défini")
        print("Créez manuellement la database avec: Nom, Téléphone, Score, Statut, Transcript, Date")
        sys.exit(1)
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Notion-Version": API_VERSION}
    
    database_data = {
        "parent": {"page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": "Voice Leads"}}],
        "properties": {
            "Nom": {"title": {}},
            "Téléphone": {"phone_number": {}},
            "Score": {"number": {"format": "number"}},
            "Statut": {"select": {"options": [
                {"name": "Nouveau", "color": "gray"},
                {"name": "En cours", "color": "yellow"},
                {"name": "Qualifié", "color": "green"},
                {"name": "Transféré", "color": "blue"},
                {"name": "Non qualifié", "color": "red"}
            ]}},
            "Transcript": {"rich_text": {}},
            "Date": {"date": {}},
            "Entreprise": {"rich_text": {}},
            "Email": {"email": {}},
            "Notes": {"rich_text": {}}
        }
    }
    
    try:
        response = httpx.post(f"{BASE_URL}/databases", headers=headers, json=database_data, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        database_id = result.get("id")
        
        print("✅ Database 'Voice Leads' créée!")
        print(f"\\n📝 ID: {database_id}")
        print(f"👉 Ajoutez à .env: NOTION_DATABASE_ID={database_id}")
        return database_id
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Crée la database Notion pour Voice Leads")
    args = parser.parse_args()
    
    if not API_KEY:
        print("❌ NOTION_API_KEY non défini")
        print("1. Allez sur https://www.notion.so/my-integrations")
        print("2. Créez une intégration")
        print("3. Ajoutez le token à .env")
        sys.exit(1)
    
    create_voice_leads_database(API_KEY, PARENT_PAGE_ID)


if __name__ == "__main__":
    main()
'''
    write_file(base_path, "setup/create_notion_db.py", notion_db_content)
    
    # setup/test_setup.py
    test_setup_content = '''"""Script de vérification de la configuration."""

import os
import asyncio
import socket
from typing import Tuple, List

import httpx
from dotenv import load_dotenv

load_dotenv()


def check_internet() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def print_success(msg): print(f"✅ {msg}")
def print_error(msg): print(f"❌ {msg}")
def print_warning(msg): print(f"⚠️  {msg}")
def print_info(msg): print(f"ℹ️  {msg}")


async def test_gemini(api_key: str) -> Tuple[bool, str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Dis 'Test OK' en français.")
        return True, f"Réponse: {response.text[:30]}"
    except Exception as e:
        return False, str(e)


async def test_notion(api_key: str) -> Tuple[bool, str]:
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Notion-Version": "2022-06-28"}
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.notion.com/v1/search", headers=headers, json={"page_size": 1}, timeout=10.0)
            return True, "API répond"
    except Exception as e:
        return False, str(e)


async def run_all_tests():
    print("="*60)
    print("🧪 TEST DE CONFIGURATION - Gradium SDR Agent")
    print("="*60 + "\\n")
    
    print("🌐 Vérification internet...")
    if check_internet():
        print_success("Connexion OK")
    else:
        print_error("Pas de connexion")
        return
    
    print("\\n🔑 Test des clés API...\\n")
    
    results = {}
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and not gemini_key.startswith("votre_"):
        print("Test Gemini...")
        results["gemini"] = await test_gemini(gemini_key)
    else:
        results["gemini"] = (False, "Non configuré")
    
    notion_key = os.getenv("NOTION_API_KEY")
    if notion_key and not notion_key.startswith("votre_"):
        print("Test Notion...")
        results["notion"] = await test_notion(notion_key)
    else:
        results["notion"] = (False, "Non configuré")
    
    print("\\n" + "="*60)
    print("📊 RÉSULTATS")
    print("="*60 + "\\n")
    
    for service, (success, message) in results.items():
        if success:
            print_success(f"{service.upper()}: {message}")
        else:
            print_error(f"{service.upper()}: {message}")
    
    successful = sum(1 for s, m in results.items() if m[0])
    print(f"\\n{successful}/{len(results)} services configurés")


def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
'''
    write_file(base_path, "setup/test_setup.py", test_setup_content)
    
    print("✅ Fichiers setup créés")


def create_test_files(base_path: Path) -> None:
    """Crée les fichiers de test."""
    
    test_content = '''"""Tests unitaires pour le module de qualification."""

import pytest
import asyncio
import sys
sys.path.insert(0, ".")

from domain.models import Conversation, Prospect, QualificationScore
from domain.qualification import RuleBasedQualifier


class TestRuleBasedQualifier:
    @pytest.fixture
    def qualifier(self):
        return RuleBasedQualifier(qualification_threshold=80)
    
    @pytest.fixture
    def empty_conversation(self):
        conv = Conversation(id="test_001", phone_number="+33612345678")
        return conv
    
    @pytest.mark.asyncio
    async def test_qualify_empty(self, qualifier, empty_conversation):
        prospect = Prospect(phone_number="+33612345678")
        score = await qualifier.qualify(empty_conversation, prospect)
        assert score.total_score == 0
        assert score.is_qualified is False
    
    @pytest.mark.asyncio
    async def test_qualify_with_budget(self, qualifier, empty_conversation):
        empty_conversation.add_message("user", "Nous avons un budget de 5000 euros")
        prospect = Prospect(phone_number="+33612345678")
        score = await qualifier.qualify(empty_conversation, prospect)
        assert score.budget_score > 0
    
    def test_score_calculation(self):
        score = QualificationScore(budget_score=20, authority_score=20, need_score=20, timeline_score=20)
        total = score.calculate_total()
        assert total == 80
        assert score.is_qualified is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    write_file(base_path, "tests/test_qualification.py", test_content)
    
    print("✅ Fichiers de test créés")


def create_config_files(base_path: Path) -> None:
    """Crée les fichiers de configuration."""
    
    env_example = '''# Configuration Gradium-SDR-Agent

APP_NAME=Gradium-SDR-Agent
APP_ENV=development
DEBUG=false
LOG_LEVEL=INFO

# Google Gemini (1M tokens/jour gratuit)
GEMINI_API_KEY=votre_cle_gemini
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.7

# Gradium (optionnel)
# GRADIUM_API_KEY=votre_cle
VOICE_PROVIDER=mock
VOICE_SPEED=1.0
VOICE_LANGUAGE=fr
VOICE_ID=nova

# SerpAPI (100 requêtes/mois gratuit)
# SERPAPI_KEY=votre_cle
QUALIFICATION_MODE=mock

# Twilio (trial 15$)
# TWILIO_ACCOUNT_SID=ACxxx
# TWILIO_AUTH_TOKEN=xxx
# TWILIO_PHONE_NUMBER=+1234567890
# TWILIO_WEBHOOK_URL=https://xxx.ngrok.io

# Notion (gratuit)
# NOTION_API_KEY=secret_xxx
# NOTION_DATABASE_ID=xxx
# NOTION_PARENT_PAGE_ID=xxx

# Ngrok
# NGROK_AUTH_TOKEN=xxx
NGROK_REGION=eu

WEBHOOK_PORT=8000
STREAMLIT_PORT=8501
QUALIFICATION_THRESHOLD=80
'''
    write_file(base_path, ".env.example", env_example)
    
    gitignore = '''# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Virtual environments
venv/
env/
ENV/
.venv/

# Environment
.env
.env.local

# Audio
audio/
*.wav
*.mp3

# IDE
.vscode/
.idea/
*.swp

# Logs
*.log
logs/

# Database
*.db
*.sqlite

# Notion temp
.notion_db_id

# pytest
.pytest_cache/
.coverage

# macOS
.DS_Store
'''
    write_file(base_path, ".gitignore", gitignore)
    
    requirements = '''# Web Framework
fastapi==0.104.1
uvicorn==0.24.0

# Dashboard
streamlit==1.28.0

# Configuration
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Téléphonie
twilio==8.10.0

# HTTP
requests==2.31.0
httpx==0.25.0

# LLM
google-generativeai==0.3.0

# Recherche
google-search-results==2.4.2

# Tunneling
pyngrok==7.0.0

# Tests
pytest==7.4.3
pytest-asyncio==0.21.1

# Utilitaires
pandas==2.1.3
'''
    write_file(base_path, "requirements.txt", requirements)
    
    print("✅ Fichiers de configuration créés")


def create_readme(base_path: Path) -> None:
    """Crée le README.md."""
    
    readme = '''# 🎙️ Gradium-SDR-Agent

Agent SDR vocal intelligent avec architecture hexagonale.

## 📋 Prérequis

Comptes gratuits à créer:
- [Google AI Studio](https://aistudio.google.com/app/apikey) - LLM
- [Twilio](https://www.twilio.com/try-twilio) - Téléphonie
- [Notion](https://www.notion.so/) - Stockage
- [Ngrok](https://ngrok.com/) - Exposition localhost

## 🚀 Installation

```bash
# 1. Cloner
cd gradium_sdr_agent

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# Éditez .env avec vos clés API

# 5. Vérification
python setup/test_setup.py
```

## ▶️ Lancement

3 terminaux nécessaires:

**Terminal 1 - Serveur Webhook:**
```bash
uvicorn interface.webhook_server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Dashboard:**
```bash
streamlit run interface/streamlit_dashboard.py
```

**Terminal 3 - Ngrok:**
```bash
ngrok http 8000
```

## 📁 Architecture

```
gradium_sdr_agent/
├── domain/           # Cœur métier (models, ports, qualification)
├── application/      # Orchestration (conversation_service)
├── infrastructure/   # Adaptateurs (config, api, telephony, storage)
├── interface/        # UI et webhooks (streamlit, fastapi)
├── setup/            # Scripts utilitaires
└── tests/            # Tests unitaires
```

## 🔧 Configuration des Clés API

### Google Gemini (Obligatoire)
1. [AI Studio](https://aistudio.google.com/app/apikey) → Create API Key
2. `GEMINI_API_KEY=votre_cle`

### Twilio (Optionnel)
1. [Twilio](https://www.twilio.com/try-twilio) → Sign up
2. Vérifiez votre numéro
3. Copiez Account SID, Auth Token, Phone Number

### Notion (Optionnel)
1. [My Integrations](https://www.notion.so/my-integrations) → New integration
2. Copiez le Internal Integration Token
3. `python setup/create_notion_db.py`

## 📝 License

MIT
'''
    write_file(base_path, "README.md", readme)
    
    print("✅ README.md créé")


if __name__ == "__main__":
    main()
