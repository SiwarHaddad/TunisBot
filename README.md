# 🏛️ Chatbot Touristique pour Tunis

## 📋 Description du Projet

Chatbot intelligent spécialisé dans le tourisme à Tunis, utilisant une **approche hybride** combinant plusieurs techniques de NLP.

### Objectifs
- Fournir des informations touristiques sur Tunis (lieux, restaurants, transports, histoire)
- Démontrer l'utilisation de différentes techniques NLP
- Comparer les approches: règles, TF-IDF, et embeddings

---

## 🔧 Techniques Utilisées

### 1. **Approche par Règles (Pattern Matching)**
- Reconnaissance de patterns avec expressions régulières
- Gestion des salutations, commandes simples
- Réponses déterministes et rapides

### 2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
- Vectorisation des questions/réponses
- Calcul de similarité cosinus
- Recherche dans la base de connaissances

### 3. **Embeddings Sémantiques (Sentence-BERT)**
- Modèle pré-entraîné multilingue: `paraphrase-multilingual-MiniLM-L12-v2`
- Représentation vectorielle dense des phrases
- Meilleure compréhension du sens

### 4. **Approche Hybride**
- Combine les 3 techniques précédentes
- Sélection intelligente de la meilleure méthode
- Fallback en cas de non-correspondance

---

## Structure du Projet

```
chatbot-tunis/
│
├── tunis_chatbot.py              # Code principal (console)
├── app.py                        # Application Flask (serveur web)
├── templates/
│   └── index.html                # Interface web connectée
├── tunis_chatbot_web.html        # Interface web standalone
├── requirements.txt              # Dépendances Python
├── README.md                     # Documentation
└── presentation/
    ├── slides.pdf                # Présentation du projet
    ├── article.pdf               # Article scientifique analysé
    └── rapport.pdf               # Rapport optionnel
```

---

## Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étape 1: Cloner ou créer le projet

```bash
mkdir chatbot-tunis
cd chatbot-tunis
```

### Étape 2: Créer un environnement virtuel (recommandé)

**Windows:**
```bash
python -m venv .venv
venv\Scripts\activate.ps1 # if it didn't work, use absolute path
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Étape 3: Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt`:**
```
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6
sentence-transformers>=2.2.0
flask>=2.0.0
flask-cors>=3.0.0
```

### Étape 4: Télécharger les ressources NLTK

Le script le fait automatiquement, mais si nécessaire:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Utilisation

### Option 1: Interface Console

```bash
python tunis_chatbot.py
```

**Exemple d'interaction:**
```
🏛️  CHATBOT TOURISTIQUE - TUNIS  
============================================================
Techniques utilisées:
  ✓ Règles (pattern matching)
  ✓ TF-IDF (similarité vectorielle)
  ✓ Embeddings (Sentence-BERT)

🤖 Bot: Bonjour! Je suis votre guide touristique virtuel...

👤 Vous: Quels sont les lieux touristiques?
🤖 Bot: Les principaux lieux touristiques à Tunis incluent:
- La Médina de Tunis (classée UNESCO)
- Le site archéologique de Carthage
...
   [Méthode: embedding (score: 0.85)]
```

### Option 2: Interface Web Standalone

Ouvrir `tunis_chatbot_web.html` directement dans un navigateur. Cette version utilise JavaScript et fonctionne sans serveur.

### Option 3: Application Web Flask (Recommandée)

**1. Créer le dossier templates:**
```bash
mkdir templates
# Copier index.html dans templates/
```

**2. Démarrer le serveur:**
```bash
python app.py
```

**3. Ouvrir dans le navigateur:**
```
http://localhost:5000
```

L'interface web offre:
- Chat interactif en temps réel
- Boutons de questions rapides
- Statistiques d'utilisation
- Indicateur de méthode utilisée
- Design moderne et responsive

---

## 📊 Base de Connaissances

Le chatbot dispose d'informations sur:

### 🏛️ Lieux Touristiques
- Médina de Tunis (UNESCO)
- Site archéologique de Carthage
- Village de Sidi Bou Saïd
- Musée National du Bardo
- Mosquée Zitouna
- Avenue Habib Bourguiba

### 🍽️ Restaurants & Gastronomie
- Recommandations de restaurants
- Spécialités tunisiennes (couscous, brik, lablabi...)
- Où manger par quartier

### 🚇 Transports
- Métro, TGM, bus, taxis
- Comment se déplacer
- Depuis l'aéroport

### 📅 Itinéraires
- Visite d'une journée
- Programme weekend
- Circuits thématiques

### 📚 Histoire & Culture
- Histoire de Carthage
- Patrimoine UNESCO
- Traditions locales

### ℹ️ Informations Pratiques
- Meilleure période pour visiter
- Hébergement
- Sécurité

---

## 🧪 Exemples de Questions

Essayez ces questions pour tester le chatbot:

```
1. "Bonjour"
2. "Quels sont les principaux lieux touristiques?"
3. "Où manger à Tunis?"
4. "Comment visiter Carthage?"
5. "Que faire en une journée?"
6. "Comment se déplacer?"
7. "Quelle est la meilleure période pour visiter?"
8. "Que voir dans la Médina?"
9. "Spécialités tunisiennes à goûter?"
10. "Au revoir"
```

---

## 📈 Évaluation des Méthodes

### Statistiques d'Utilisation

Consultez `/api/stats` pour voir:
- Nombre total de messages
- Distribution des méthodes utilisées
- Performance du système

---

## 🔍 Fonctionnalités Avancées

### API Flask Endpoints

#### `POST /api/chat`
Envoyer un message au chatbot
```json
{
  "message": "Où manger à Tunis?"
}
```

Réponse:
```json
{
  "success": true,
  "response": "Bonnes adresses à Tunis:\n- Dar El Jeld...",
  "method": "embedding (score: 0.87)"
}
```

#### `GET /api/stats`
Obtenir les statistiques
```json
{
  "success": true,
  "total_messages": 25,
  "methods_distribution": {
    "rule-based": 4,
    "tfidf": 8,
    "embedding": 13
  }
}
```

#### `POST /api/reset`
Réinitialiser la conversation

#### `GET /api/history`
Obtenir l'historique complet

---

## 🐛 Dépannage

### Problème: sentence-transformers ne s'installe pas

**Solution:** Le chatbot fonctionne sans embeddings, utilisant uniquement règles + TF-IDF
```bash
# Si l'installation échoue, continuez quand même
# Le chatbot détectera l'absence et s'adaptera
```

### Problème: Erreur NLTK "punkt not found"

**Solution:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Problème: Flask ne démarre pas

**Solution:**
```bash
pip install flask flask-cors
```

### Problème: Encodage des caractères

**Solution:** S'assurer que tous les fichiers sont en UTF-8
```python
# Ajouter en haut du fichier Python:
# -*- coding: utf-8 -*-
```

---

## 📚 Améliorations Possibles

### Court Terme
- ✅ Ajouter plus de données dans la base de connaissances
- ✅ Implémenter la gestion du contexte conversationnel
- ✅ Ajouter des images et cartes interactives
- ✅ Support multilingue (arabe, anglais)

### Long Terme
- 🔄 Intégration avec des APIs externes (météo, réservations)
- 🔄 Utilisation de modèles de langage plus avancés (GPT)
- 🔄 Apprentissage par renforcement
- 🔄 Interface vocale

---

## 👥 Contributeurs

Mini-projet TALN 2025-2026
- Trinôme: Siwar Haddad - Oumayma Hammami - Oussama Chaabane
- Établissement: ENSI (M2-SS AIS)

---

## 📄 Licence

Ce projet est développé dans un cadre académique.

---

**Bon voyage à Tunis! 🇹🇳🏛️**