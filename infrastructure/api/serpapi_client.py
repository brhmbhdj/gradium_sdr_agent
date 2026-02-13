"""Client SerpAPI - Recherche web pour qualification avancée."""

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
