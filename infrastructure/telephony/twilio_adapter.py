"""Adaptateur Twilio - Implémentation du TelephonyPort."""

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
