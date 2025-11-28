import streamlit as st
import joblib
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Téléchargement des ressources NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    .spam-box {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .ham-box {
        background-color: #D5F4E6;
        border-left: 5px solid #2ECC71;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Fonction de prétraitement
@st.cache_resource
def load_preprocessing_tools():
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    return stemmer, stop_words

stemmer, stop_words = load_preprocessing_tools()

def preprocess_text(text):
    """Prétraitement du texte"""
    # Conversion en minuscules
    text = text.lower()

    # Suppression de la ponctuation et caractères spéciaux
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenisation
    tokens = word_tokenize(text)

    # Suppression des stopwords et stemming
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return ' '.join(processed_tokens)

# Chargement du modèle et du vectorizer
@st.cache_resource
def load_model():
    try:
        # Charger les métadonnées
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)

        model_name = metadata['model_name'].replace(' ', '_').lower()

        # Charger le modèle
        model = joblib.load(f'models/best_spam_detector_{model_name}.pkl')

        # Charger le vectorizer
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

        return model, vectorizer, metadata
    except FileNotFoundError:
        st.error("⚠️ Modèle non trouvé. Veuillez d'abord entraîner le modèle en exécutant le notebook.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Erreur lors du chargement du modèle: {str(e)}")
        st.stop()

model, vectorizer, metadata = load_model()

# Fonction de prédiction
def predict_email(subject, body):
    """Prédit si un email est spam ou ham"""
    # Combiner subject et body (comme dans le training)
    combined_text = subject + ' ' + body

    # Prétraitement
    processed = preprocess_text(combined_text)

    # Vectorisation
    vectorized = vectorizer.transform([processed])

    # Prédiction
    prediction = model.predict(vectorized)[0]

    # Probabilité
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vectorized)[0]
        confidence = proba[prediction]
    else:
        # Pour SVM, utiliser decision_function
        decision = model.decision_function(vectorized)[0]
        confidence = 1 / (1 + abs(decision))  # Approximation

    return prediction, confidence

# Header
st.markdown('<div class="main-header">📧 Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analysez un email pour détecter s\'il s\'agit d\'un spam</div>', unsafe_allow_html=True)

# Champs de saisie
email_subject = st.text_input(
    "Sujet de l'email:",
    placeholder="Entrez le sujet de l'email..."
)

email_body = st.text_area(
    "Corps de l'email:",
    height=200,
    placeholder="Entrez le contenu de l'email..."
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_button = st.button("🔍 Analyser l'Email", use_container_width=True, type="primary")

if analyze_button:
    if email_subject.strip() == "" and email_body.strip() == "":
        st.warning("⚠️ Veuillez entrer au moins un sujet ou un corps d'email.")
    else:
        with st.spinner("Analyse en cours..."):
            # Prédiction
            prediction, confidence = predict_email(email_subject, email_body)

            # Affichage du résultat
            st.divider()

            if prediction == 1:  # Spam
                st.markdown(f"""
                <div class="spam-box">
                    <h2 style="color: #C0392B; margin: 0;">⚠️ SPAM DÉTECTÉ</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px;">
                        Cet email a été classifié comme <strong>SPAM</strong> avec une confiance de <strong>{confidence*100:.1f}%</strong>
                    </p>
                    <p style="color: #666; margin-top: 10px;">
                        ⚡ Action recommandée: Ne pas ouvrir les liens et supprimer cet email.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge de confiance
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': "Niveau de Confiance"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#E74C3C"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FADBD8"},
                            {'range': [50, 75], 'color': "#F5B7B1"},
                            {'range': [75, 100], 'color': "#E74C3C"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            else:  # Ham
                st.markdown(f"""
                <div class="ham-box">
                    <h2 style="color: #27AE60; margin: 0;">✅ EMAIL LÉGITIME</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px;">
                        Cet email a été classifié comme <strong>LÉGITIME</strong> avec une confiance de <strong>{confidence*100:.1f}%</strong>
                    </p>
                    <p style="color: #666; margin-top: 10px;">
                        ✓ Cet email semble sûr et peut être lu normalement.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge de confiance
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': "Niveau de Confiance"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ECC71"},
                        'steps': [
                            {'range': [0, 50], 'color': "#D5F4E6"},
                            {'range': [50, 75], 'color': "#A9DFBF"},
                            {'range': [75, 100], 'color': "#2ECC71"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Affichage du texte prétraité
            with st.expander("🔬 Voir le texte prétraité"):
                combined = email_subject + ' ' + email_body
                processed = preprocess_text(combined)
                st.code(processed)
                st.caption(f"Nombre de mots après prétraitement: {len(processed.split())}")
