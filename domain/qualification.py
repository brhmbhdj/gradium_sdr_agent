"""Logique métier de qualification des leads."""

from typing import Dict, List, Any, Optional

from .models import Conversation, Prospect, QualificationScore
from .ports import QualificationPort


class RuleBasedQualifier(QualificationPort):
    """Qualificateur basé sur des règles simples."""
    
    KEYWORDS = {
        "budget": ["budget", "euros", "€", "dollars", "$", "investir", "investissement", "prix", "coût", "tarif", "combien"],
        "authority": ["décide", "décision", "décideur", "responsable", "directeur", "manager", "chef", "ceo", "fondateur"],
        "need": ["besoin", "problème", "solution", "cherchons", "intéressé", "améliorer", "optimiser"],
        "timeline": ["délai", "timeline", "quand", "date", "mois", "semaine", "urgent", "rapidement"],
        "transfer": ["transfert", "transférer", "parler à", "humain", "conseiller", "commercial"]
    }
    
    def __init__(self, qualification_threshold: int = 80):
        self.qualification_threshold = qualification_threshold
    
    def _analyze_category(self, conversation: Conversation, category: str) -> tuple:
        keywords = self.KEYWORDS.get(category, [])
        found_keywords = []
        transcript = conversation.get_transcript().lower()
        
        for keyword in keywords:
            if keyword.lower() in transcript:
                found_keywords.append(keyword)
        
        score = min(len(found_keywords) * 5, 25)
        return score, found_keywords
    
    async def qualify(self, conversation: Conversation, prospect: Prospect) -> QualificationScore:
        score = QualificationScore()
        details = {}
        
        score.budget_score, details["budget_keywords"] = self._analyze_category(conversation, "budget")
        score.authority_score, details["authority_keywords"] = self._analyze_category(conversation, "authority")
        score.need_score, details["need_keywords"] = self._analyze_category(conversation, "need")
        score.timeline_score, details["timeline_keywords"] = self._analyze_category(conversation, "timeline")
        
        score.calculate_total()
        score.details = details
        
        return score
    
    async def should_transfer(self, conversation: Conversation, score: QualificationScore) -> bool:
        if score.total_score >= self.qualification_threshold:
            return True
        
        transcript = conversation.get_transcript().lower()
        transfer_keywords = self.KEYWORDS.get("transfer", [])
        
        for keyword in transfer_keywords:
            if keyword.lower() in transcript:
                return True
        
        return False


class ResearchBasedQualifier(RuleBasedQualifier):
    """Qualificateur enrichi par la recherche web."""
    
    def __init__(self, qualification_threshold: int = 80, serpapi_client: Optional[Any] = None):
        super().__init__(qualification_threshold)
        self.serpapi_client = serpapi_client
    
    async def research_company(self, company_name: str) -> Dict[str, Any]:
        if not self.serpapi_client:
            return {"error": "SerpAPI client not configured"}
        
        try:
            results = await self.serpapi_client.search(query=f"{company_name} entreprise", num_results=5)
            return {"company": company_name, "results": results}
        except Exception as e:
            return {"error": str(e), "company": company_name}
    
    async def qualify(self, conversation: Conversation, prospect: Prospect) -> QualificationScore:
        score = await super().qualify(conversation, prospect)
        
        if prospect.company and self.serpapi_client:
            research_data = await self.research_company(prospect.company)
            score.details["research"] = research_data
            
            if "results" in research_data and len(research_data["results"]) > 0:
                score.authority_score = min(score.authority_score + 5, 25)
                score.details["company_verified"] = True
        
        score.calculate_total()
        return score
