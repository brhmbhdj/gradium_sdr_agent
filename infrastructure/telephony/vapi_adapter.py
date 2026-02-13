"""Adaptateur VAPI - Alternative à Twilio."""

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
