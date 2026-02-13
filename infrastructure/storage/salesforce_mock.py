"""Mock Salesforce - Implémentation mock du StoragePort."""

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
