"""Script de vérification de la configuration."""

import os
import asyncio
import socket
from typing import Tuple, List

import httpx
from dotenv import load_dotenv

load_dotenv()


def check_internet() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def print_success(msg): print(f"✅ {msg}")
def print_error(msg): print(f"❌ {msg}")
def print_warning(msg): print(f"⚠️  {msg}")
def print_info(msg): print(f"ℹ️  {msg}")


async def test_gemini(api_key: str) -> Tuple[bool, str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Dis 'Test OK' en français.")
        return True, f"Réponse: {response.text[:30]}"
    except Exception as e:
        return False, str(e)


async def test_notion(api_key: str) -> Tuple[bool, str]:
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Notion-Version": "2022-06-28"}
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.notion.com/v1/search", headers=headers, json={"page_size": 1}, timeout=10.0)
            return True, "API répond"
    except Exception as e:
        return False, str(e)


async def run_all_tests():
    print("="*60)
    print("🧪 TEST DE CONFIGURATION - Gradium SDR Agent")
    print("="*60 + "\n")
    
    print("🌐 Vérification internet...")
    if check_internet():
        print_success("Connexion OK")
    else:
        print_error("Pas de connexion")
        return
    
    print("\n🔑 Test des clés API...\n")
    
    results = {}
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and not gemini_key.startswith("votre_"):
        print("Test Gemini...")
        results["gemini"] = await test_gemini(gemini_key)
    else:
        results["gemini"] = (False, "Non configuré")
    
    notion_key = os.getenv("NOTION_API_KEY")
    if notion_key and not notion_key.startswith("votre_"):
        print("Test Notion...")
        results["notion"] = await test_notion(notion_key)
    else:
        results["notion"] = (False, "Non configuré")
    
    print("\n" + "="*60)
    print("📊 RÉSULTATS")
    print("="*60 + "\n")
    
    for service, (success, message) in results.items():
        if success:
            print_success(f"{service.upper()}: {message}")
        else:
            print_error(f"{service.upper()}: {message}")
    
    successful = sum(1 for s, m in results.items() if m[0])
    print(f"\n{successful}/{len(results)} services configurés")


def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
