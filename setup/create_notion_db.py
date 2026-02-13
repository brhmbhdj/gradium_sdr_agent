"""Script de création de la database Notion."""

import os
import sys
import argparse
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NOTION_API_KEY")
PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID")
BASE_URL = "https://api.notion.com/v1"
API_VERSION = "2022-06-28"


def create_voice_leads_database(api_key: str, parent_page_id: Optional[str] = None) -> str:
    if not parent_page_id:
        print("⚠️  NOTION_PARENT_PAGE_ID non défini")
        print("Créez manuellement la database avec: Nom, Téléphone, Score, Statut, Transcript, Date")
        sys.exit(1)
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Notion-Version": API_VERSION}
    
    database_data = {
        "parent": {"page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": "Voice Leads"}}],
        "properties": {
            "Nom": {"title": {}},
            "Téléphone": {"phone_number": {}},
            "Score": {"number": {"format": "number"}},
            "Statut": {"select": {"options": [
                {"name": "Nouveau", "color": "gray"},
                {"name": "En cours", "color": "yellow"},
                {"name": "Qualifié", "color": "green"},
                {"name": "Transféré", "color": "blue"},
                {"name": "Non qualifié", "color": "red"}
            ]}},
            "Transcript": {"rich_text": {}},
            "Date": {"date": {}},
            "Entreprise": {"rich_text": {}},
            "Email": {"email": {}},
            "Notes": {"rich_text": {}}
        }
    }
    
    try:
        response = httpx.post(f"{BASE_URL}/databases", headers=headers, json=database_data, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        database_id = result.get("id")
        
        print("✅ Database 'Voice Leads' créée!")
        print(f"\n📝 ID: {database_id}")
        print(f"👉 Ajoutez à .env: NOTION_DATABASE_ID={database_id}")
        return database_id
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Crée la database Notion pour Voice Leads")
    args = parser.parse_args()
    
    if not API_KEY:
        print("❌ NOTION_API_KEY non défini")
        print("1. Allez sur https://www.notion.so/my-integrations")
        print("2. Créez une intégration")
        print("3. Ajoutez le token à .env")
        sys.exit(1)
    
    create_voice_leads_database(API_KEY, PARENT_PAGE_ID)


if __name__ == "__main__":
    main()
