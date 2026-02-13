"""Service d'orchestration des conversations."""

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
