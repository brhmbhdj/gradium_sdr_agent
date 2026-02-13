"""Adaptateur Google Gemini - Implémentation du LLMPort."""

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
        return "\n".join(messages)
    
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
