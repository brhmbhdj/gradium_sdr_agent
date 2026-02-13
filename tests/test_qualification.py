"""Tests unitaires pour le module de qualification."""

import pytest
import asyncio
import sys
sys.path.insert(0, ".")

from domain.models import Conversation, Prospect, QualificationScore
from domain.qualification import RuleBasedQualifier


class TestRuleBasedQualifier:
    @pytest.fixture
    def qualifier(self):
        return RuleBasedQualifier(qualification_threshold=80)
    
    @pytest.fixture
    def empty_conversation(self):
        conv = Conversation(id="test_001", phone_number="+33612345678")
        return conv
    
    @pytest.mark.asyncio
    async def test_qualify_empty(self, qualifier, empty_conversation):
        prospect = Prospect(phone_number="+33612345678")
        score = await qualifier.qualify(empty_conversation, prospect)
        assert score.total_score == 0
        assert score.is_qualified is False
    
    @pytest.mark.asyncio
    async def test_qualify_with_budget(self, qualifier, empty_conversation):
        empty_conversation.add_message("user", "Nous avons un budget de 5000 euros")
        prospect = Prospect(phone_number="+33612345678")
        score = await qualifier.qualify(empty_conversation, prospect)
        assert score.budget_score > 0
    
    def test_score_calculation(self):
        score = QualificationScore(budget_score=20, authority_score=20, need_score=20, timeline_score=20)
        total = score.calculate_total()
        assert total == 80
        assert score.is_qualified is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
