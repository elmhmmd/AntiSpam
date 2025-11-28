# Système de Détection de Spam

Système intelligent de détection de spams utilisant des techniques de NLP

## Installation

1. Installer les dépendances:
```bash
pip install -r requirements.txt
```

2. Entraîner le modèle:
```bash
# Ouvrir et exécuter antispam.ipynb dans Jupyter
jupyter notebook antispam.ipynb
```

3. Lancer l'application Streamlit:
```bash
streamlit run app.py
```

## Structure du Projet

```
Anti-Spam/
├── antispam.ipynb          # Notebook d'entraînement du modèle
├── app.py                  # Application Streamlit
├── DataSet_Emails.csv      # Dataset d'entraînement
├── requirements.txt        # Dépendances Python
├── models/                 # Modèles sauvegardés (après entraînement)
│   ├── best_spam_detector_*.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json
└── README.md             
```

## Fonctionnalités

- ✅ Détection automatique de spams
- ✅ Analyse en temps réel
- ✅ Score de confiance
- ✅ Prétraitement NLP avancé
- ✅ Plusieurs modèles ML testés
- ✅ Interface utilisateur intuitive
- ✅ Visualisations interactives

## Technologies Utilisées

- **UI:** Streamlit
- **ML:** Scikit-learn (Naive Bayes, Logistic Regression, Random Forest, SVM)
- **NLP:** NLTK (Tokenization, Stemming, Stopwords)
- **Vectorisation:** TF-IDF
- **Visualisation:** Plotly, Seaborn, Matplotlib

## Pipeline de Traitement

1. **Normalisation:** Conversion en minuscules
2. **Nettoyage:** Suppression de la ponctuation et caractères spéciaux
3. **Tokenisation:** Découpage en mots individuels
4. **Filtrage:** Suppression des stopwords
5. **Stemming:** Réduction des mots à leur racine (PorterStemmer)
6. **Vectorisation:** TF-IDF avec n-grams (1,2)
7. **Prédiction:** Classification avec le meilleur modèle
8. **Résultat:** Spam (malveillant) ou Ham (légitime) avec score de confiance

## Utilisation

### Via l'Application Web

1. Lancez l'application: `streamlit run app.py`
2. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8501)
3. Collez le texte de l'email dans la zone de texte
4. Cliquez sur "Analyser l'Email"
5. Consultez le résultat et le score de confiance

### Via Code Python

```python
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Charger le modèle et le vectorizer
model = joblib.load('models/best_spam_detector_*.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Prétraiter le texte
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return ' '.join(processed_tokens)

# Faire une prédiction
email_text = "Your text here"
processed = preprocess_text(email_text)
vectorized = vectorizer.transform([processed])
prediction = model.predict(vectorized)[0]

print("SPAM" if prediction == 1 else "HAM")
```

## Performances du Modèle

Les performances varient selon le modèle choisi. Le meilleur modèle est sélectionné automatiquement lors de l'entraînement basé sur le F1-Score.

Métriques typiques:
- **Accuracy:** > 95%
- **Precision:** > 95%
- **Recall:** > 90%
- **F1-Score:** > 92%