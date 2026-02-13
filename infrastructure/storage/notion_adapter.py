"""Adaptateur Notion - Implémentation du StoragePort."""

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
