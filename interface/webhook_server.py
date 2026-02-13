"""Serveur Webhook FastAPI."""

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
