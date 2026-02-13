# 🎙️ Gradium-SDR-Agent

Agent SDR vocal intelligent avec architecture hexagonale.

## 📋 Prérequis

Comptes gratuits à créer:
- [Google AI Studio](https://aistudio.google.com/app/apikey) - LLM
- [Twilio](https://www.twilio.com/try-twilio) - Téléphonie
- [Notion](https://www.notion.so/) - Stockage
- [Ngrok](https://ngrok.com/) - Exposition localhost

## 🚀 Installation

```bash
# 1. Cloner
cd gradium_sdr_agent

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# Éditez .env avec vos clés API

# 5. Vérification
python setup/test_setup.py
```

## ▶️ Lancement

3 terminaux nécessaires:

**Terminal 1 - Serveur Webhook:**
```bash
uvicorn interface.webhook_server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Dashboard:**
```bash
streamlit run interface/streamlit_dashboard.py
```

**Terminal 3 - Ngrok:**
```bash
ngrok http 8000
```

## 📁 Architecture

```
gradium_sdr_agent/
├── domain/           # Cœur métier (models, ports, qualification)
├── application/      # Orchestration (conversation_service)
├── infrastructure/   # Adaptateurs (config, api, telephony, storage)
├── interface/        # UI et webhooks (streamlit, fastapi)
├── setup/            # Scripts utilitaires
└── tests/            # Tests unitaires
```

## 🔧 Configuration des Clés API

### Google Gemini (Obligatoire)
1. [AI Studio](https://aistudio.google.com/app/apikey) → Create API Key
2. `GEMINI_API_KEY=votre_cle`

### Twilio (Optionnel)
1. [Twilio](https://www.twilio.com/try-twilio) → Sign up
2. Vérifiez votre numéro
3. Copiez Account SID, Auth Token, Phone Number

### Notion (Optionnel)
1. [My Integrations](https://www.notion.so/my-integrations) → New integration
2. Copiez le Internal Integration Token
3. `python setup/create_notion_db.py`

## 📝 License

MIT
