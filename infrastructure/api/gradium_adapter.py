"""Adaptateur Gradium - Implémentation du VoicePort."""

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
        mock_audio = b"\xff\xf3\x44\xc0" + b"\x00" * 417
        return SynthesisResult(audio_data=mock_audio, duration_seconds=len(text) * 0.08, format="mp3")
