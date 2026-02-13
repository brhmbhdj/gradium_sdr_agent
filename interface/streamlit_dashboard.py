"""Dashboard Streamlit - Interface de monitoring."""

import logging
from datetime import datetime

import streamlit as st
import pandas as pd

import sys
sys.path.insert(0, ".")

from infrastructure.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Gradium SDR Agent", page_icon="🎙️", layout="wide")


def main():
    st.title("🎙️ Gradium SDR Agent")
    st.markdown("### Dashboard de monitoring temps réel")
    st.divider()
    
    # Statut des services
    st.subheader("🔌 Statut des Services")
    status = settings.get_status()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🗣️ Voix**")
        st.success("✅ Gradium") if status["gradium"] else st.warning("⚠️ Mode Mock")
    with col2:
        st.markdown("**🤖 LLM**")
        st.success("✅ Gemini") if status["gemini"] else st.warning("⚠️ Mode Mock")
    with col3:
        st.markdown("**📞 Téléphonie**")
        st.success("✅ Configuré") if status["twilio"] else st.error("❌ Non configuré")
    
    st.divider()
    
    # Statistiques
    st.subheader("📈 Statistiques")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Appels aujourd'hui", "12", "+3")
    c2.metric("Leads qualifiés", "5", "+2")
    c3.metric("Taux de qualification", "42%", "+5%")
    c4.metric("Score moyen", "68", "+3")
    
    st.divider()
    
    # Conversations actives
    st.subheader("📞 Conversations Actives")
    demo_data = [
        {"ID": "conv_001", "Téléphone": "+33 6 12 34 56 78", "Durée": "2:34", "Messages": 5, "Score": 65, "Statut": "En cours"},
        {"ID": "conv_002", "Téléphone": "+33 7 98 76 54 32", "Durée": "1:12", "Messages": 3, "Score": 45, "Statut": "En cours"}
    ]
    df = pd.DataFrame(demo_data)
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    # Leads qualifiés
    st.subheader("⭐ Leads Qualifiés")
    demo_leads = [
        {"Nom": "Jean Dupont", "Entreprise": "TechCorp", "Téléphone": "+33 6 11 22 33 44", "Score": 85, "Statut": "Transféré"},
        {"Nom": "Marie Martin", "Entreprise": "StartupXYZ", "Téléphone": "+33 7 55 66 77 88", "Score": 92, "Statut": "Qualifié"}
    ]
    df_leads = pd.DataFrame(demo_leads)
    st.dataframe(df_leads, use_container_width=True)
    
    st.divider()
    st.caption("Gradium-SDR-Agent v1.0")


if __name__ == "__main__":
    main()
