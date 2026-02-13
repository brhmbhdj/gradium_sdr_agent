"""Modèles de domaine pour Gradium-SDR-Agent."""

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
        return "\n".join(lines)
    
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
