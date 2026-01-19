"""
Chatbot Touristique pour Tunis - Approche Hybride
Utilise: Règles, TF-IDF, Embeddings (Sentence-BERT)
"""

import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Téléchargement des ressources NLTK (à faire une seule fois)
def download_nltk_resources():
    resources = {
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab', # Ressource manquante provoquant l'erreur
        'corpora/stopwords': 'stopwords'
    }
    
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Téléchargement de la ressource NLTK : {package}...")
            nltk.download(package)

download_nltk_resources()

# Pour utiliser Sentence-BERT, installer: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    USE_EMBEDDINGS = True
except ImportError:
    print("⚠️ sentence-transformers non installé. Utilisation de TF-IDF uniquement.")
    print("Pour installer: pip install sentence-transformers")
    USE_EMBEDDINGS = False


class TunisChatbot:
    def __init__(self):
        self.stop_words = set(stopwords.words('french'))
        self.knowledge_base = self._load_knowledge_base()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.conversation_history = []
        
        # Préparation TF-IDF
        self.questions = [item['question'] for item in self.knowledge_base]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.questions)
        
        # Chargement du modèle d'embeddings (si disponible)
        if USE_EMBEDDINGS:
            print("🔄 Chargement du modèle Sentence-BERT...")
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.question_embeddings = self.sentence_model.encode(self.questions)
            print("✅ Modèle chargé avec succès!")
        else:
            self.sentence_model = None
            self.question_embeddings = None
        
        # Règles de pattern matching
        self.patterns = {
            'salutation': [r'\b(bonjour|salut|hey|hello|bonsoir)\b'],
            'au_revoir': [r'\b(au revoir|bye|à bientôt|merci|adieu)\b'],
            'aide': [r'\b(aide|aider|comment|commencer|que faire)\b'],
            'nom': [r'\b(qui es-tu|ton nom|tu es qui|c\'est quoi ton nom)\b'],
        }
    
    def _load_knowledge_base(self):
        """Base de connaissances riche sur Tunis"""
        return [
            # LIEUX TOURISTIQUES
            {
                "question": "Quels sont les principaux lieux touristiques à Tunis?",
                "answer": "Les principaux lieux touristiques à Tunis incluent:\n- La Médina de Tunis (classée UNESCO)\n- Le site archéologique de Carthage\n- Le village de Sidi Bou Saïd\n- Le Musée National du Bardo\n- La Mosquée Zitouna\n- L'Avenue Habib Bourguiba",
                "category": "lieux"
            },
            {
                "question": "Que voir dans la Médina de Tunis?",
                "answer": "Dans la Médina de Tunis, vous pouvez visiter:\n- La Mosquée Zitouna (la plus grande mosquée de Tunis)\n- Les souks traditionnels (parfums, tissus, bijoux)\n- Dar Lasram et autres palais ottomans\n- Les médersas historiques\n- Les portes anciennes (Bab El Bhar, Bab Souika)\nC'est un labyrinthe fascinant de ruelles étroites!",
                "category": "lieux"
            },
            {
                "question": "Comment visiter Carthage?",
                "answer": "Pour visiter Carthage:\n- Prenez le TGM (train) depuis Tunis Marine jusqu'à Carthage Hannibal\n- Achetez un billet global pour tous les sites (environ 12 DT)\n- Sites principaux: Thermes d'Antonin, Théâtre romain, Tophet, Musée de Carthage\n- Comptez une demi-journée à une journée complète\n- Combinez avec Sidi Bou Saïd tout proche!",
                "category": "lieux"
            },
            {
                "question": "Pourquoi visiter Sidi Bou Saïd?",
                "answer": "Sidi Bou Saïd est célèbre pour:\n- Ses maisons blanches et bleues iconiques\n- Ses ruelles pavées pittoresques\n- La vue panoramique sur la Méditerranée\n- Le Café des Nattes (thé à la menthe et pignons)\n- Les galeries d'art et boutiques d'artisanat\n- L'ambiance bohème et artistique\nC'est l'un des plus beaux villages de Tunisie!",
                "category": "lieux"
            },
            {
                "question": "Que voir au Musée du Bardo?",
                "answer": "Le Musée National du Bardo abrite:\n- La plus grande collection de mosaïques romaines au monde\n- Des antiquités puniques et romaines\n- Des collections islamiques\n- Le célèbre baptistère de Dougga\n- Architecture magnifique dans un ancien palais beylical\nComptez 2-3 heures pour la visite. Fermé le lundi.",
                "category": "lieux"
            },
            
            # RESTAURANTS ET GASTRONOMIE
            {
                "question": "Où manger à Tunis?",
                "answer": "Bonnes adresses à Tunis:\n- Dar El Jeld (cuisine traditionnelle raffinée, Médina)\n- Le Baroque (cuisine fusion, La Marsa)\n- Chez Slah (poissons, La Goulette)\n- La Closerie (française, Lac de Tunis)\n- M'rabet (pâtisseries, Avenue Habib Bourguiba)\n- Essaraya (traditionnel, Gammarth)",
                "category": "restaurants"
            },
            {
                "question": "Quelles spécialités tunisiennes goûter?",
                "answer": "Spécialités incontournables:\n- Couscous (vendredi tradition)\n- Brik à l'œuf (feuille croustillante)\n- Tajine tunisien (différent du marocain)\n- Ojja (plat aux œufs épicé)\n- Lablabi (soupe de pois chiches)\n- Mechouia (salade grillée)\n- Makroudh et baklawa (pâtisseries)\n- Thé à la menthe et pignons",
                "category": "gastronomie"
            },
            {
                "question": "Où manger des bons bricks?",
                "answer": "Pour déguster d'excellents bricks:\n- M'rabet (Avenue Habib Bourguiba)\n- Dans les petits restaurants de la Médina\n- Chez Slah à La Goulette\n- Au marché central\nLe brik à l'œuf est le plus populaire, mais il existe aussi au thon, aux crevettes, et à la viande.",
                "category": "restaurants"
            },
            
            # TRANSPORTS
            {
                "question": "Comment se déplacer à Tunis?",
                "answer": "Moyens de transport à Tunis:\n- Métro léger (5 lignes, bon marché)\n- TGM: train de banlieue vers La Marsa/Carthage\n- Bus: réseau étendu mais souvent bondé\n- Taxis: jaunes (compteur) ou louages blancs (collectifs)\n- Uber et Bolt: disponibles\n- Location de voiture: pour plus de liberté\nLe métro est le plus pratique pour le centre-ville.",
                "category": "transport"
            },
            {
                "question": "Comment aller de l'aéroport au centre-ville?",
                "answer": "De l'aéroport Tunis-Carthage au centre:\n- Taxi officiel: 10-15 DT (négociez avant), 20-30 min\n- Uber/Bolt: environ 10 DT\n- Bus ligne 35: très économique mais lent\n- Navette privée: réserver à l'avance\nL'aéroport est à seulement 8 km du centre-ville.",
                "category": "transport"
            },
            
            # HISTOIRE ET CULTURE
            {
                "question": "Quelle est l'histoire de Carthage?",
                "answer": "Carthage, fondée par les Phéniciens en 814 av. J.-C., fut:\n- Une puissante cité-état maritime et commerciale\n- Rivale de Rome (Guerres puniques)\n- Patrie du célèbre général Hannibal\n- Détruite par Rome en 146 av. J.-C.\n- Reconstruite comme capitale romaine d'Afrique\n- Aujourd'hui site archéologique UNESCO\nUne histoire de 3000 ans!",
                "category": "histoire"
            },
            {
                "question": "Pourquoi la Médina est-elle classée UNESCO?",
                "answer": "La Médina de Tunis est classée UNESCO car:\n- Fondée au 7ème siècle (époque islamique)\n- Architecture arabo-musulmane préservée\n- Plus de 700 monuments historiques\n- Souks et artisanat traditionnel vivant\n- Exemple exceptionnel de ville arabe médiévale\n- Centre culturel et religieux important\nC'est un patrimoine mondial depuis 1979.",
                "category": "histoire"
            },
            
            # INFORMATIONS PRATIQUES
            {
                "question": "Quelle est la meilleure période pour visiter Tunis?",
                "answer": "Meilleures périodes pour visiter Tunis:\n- Printemps (mars-mai): temps doux, 18-25°C, idéal\n- Automne (septembre-novembre): agréable, moins de touristes\n- Été (juin-août): chaud (30-35°C), animation, plages\n- Hiver (décembre-février): doux mais pluvieux\nÉvitez juillet-août si vous n'aimez pas la chaleur intense.",
                "category": "pratique"
            },
            {
                "question": "Où dormir à Tunis?",
                "answer": "Options d'hébergement:\n- Centre-ville: proche attractions, vie urbaine\n- La Marsa/Gammarth: bord de mer, calme, résidentiel\n- Sidi Bou Saïd: charme, vue, romantique\n- Médina: authentique, riads traditionnels\nBudget: auberges 15-30€, hôtels moyens 40-80€, luxe 100€+\nRéservez à l'avance en haute saison!",
                "category": "pratique"
            },
            {
                "question": "Tunis est-elle sûre pour les touristes?",
                "answer": "Tunis est généralement sûre pour les touristes:\n- Centre-ville et zones touristiques bien sécurisés\n- Précautions habituelles: attention pickpockets (Médina, transports)\n- Éviter ruelles isolées la nuit\n- Respecter les coutumes locales\n- Police touristique disponible\nLes Tunisiens sont accueillants et hospitaliers!",
                "category": "pratique"
            },
            
            # ITINÉRAIRES
            {
                "question": "Que faire en une journée à Tunis?",
                "answer": "Itinéraire d'une journée:\nMatin:\n- Médina de Tunis et Mosquée Zitouna (2h)\n- Souks et shopping artisanal (1h)\nMidi:\n- Déjeuner dans la Médina\nAprès-midi:\n- Musée du Bardo (2h)\n- Avenue Habib Bourguiba (balade)\nSoir:\n- Dîner à Sidi Bou Saïd + coucher de soleil\nAlternative: remplacer Bardo par Carthage",
                "category": "itineraire"
            },
            {
                "question": "Que faire en un weekend à Tunis?",
                "answer": "Programme weekend (2-3 jours):\nJour 1:\n- Matin: Médina + Mosquée Zitouna\n- Après-midi: Musée du Bardo\n- Soir: Avenue Bourguiba\n\nJour 2:\n- Matin: Site de Carthage (ruines romaines)\n- Après-midi: Sidi Bou Saïd (village bleu et blanc)\n- Soir: Dîner fruits de mer à La Goulette\n\nJour 3 (optionnel):\n- Plage à Gammarth ou La Marsa\n- Shopping souvenirs",
                "category": "itineraire"
            },
            
            # SHOPPING
            {
                "question": "Que ramener de Tunis comme souvenir?",
                "answer": "Souvenirs typiques de Tunis:\n- Poterie et céramique de Nabeul\n- Tapis et kilims berbères\n- Chéchia (chapeau traditionnel rouge)\n- Bijoux en argent\n- Huile d'olive tunisienne\n- Épices (harissa, ras el hanout)\n- Savon d'Alep et huile d'argan\n- Cuir et babouches\nMarchandez dans les souks (30-50% du prix initial)!",
                "category": "shopping"
            },
            
            # PLAGES
            {
                "question": "Où aller à la plage près de Tunis?",
                "answer": "Plages proches de Tunis:\n- Gammarth: plage propre, restaurants, clubs privés\n- La Marsa: populaire, ambiance familiale\n- Carthage: petites criques tranquilles\n- Raoued: plus sauvage, moins fréquentée\n- Hammamet: à 1h, stations balnéaires\nL'eau est chaude de juin à septembre (22-26°C).",
                "category": "plages"
            }
        ]
    
    def preprocess_text(self, text):
        """Prétraitement du texte"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text, language='french')
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)
    
    def rule_based_response(self, user_input):
        """Approche 1: Réponses basées sur des règles (pattern matching)"""
        user_input_lower = user_input.lower()
        
        for pattern_name, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    if pattern_name == 'salutation':
                        return "Bonjour! Je suis votre guide touristique virtuel pour Tunis. Comment puis-je vous aider à découvrir notre belle ville?"
                    elif pattern_name == 'au_revoir':
                        return "Au revoir! J'espère que vous passerez un merveilleux séjour à Tunis. Bon voyage! 🌟"
                    elif pattern_name == 'aide':
                        return "Je peux vous aider avec:\n- Les lieux touristiques (Médina, Carthage, Sidi Bou Saïd...)\n- Les restaurants et spécialités culinaires\n- Les transports et infos pratiques\n- L'histoire et la culture\n- Des itinéraires suggérés\n\nPosez-moi une question!"
                    elif pattern_name == 'nom':
                        return "Je suis TunisBot, votre assistant touristique intelligent pour découvrir Tunis et ses merveilles! 🇹🇳"
        return None
    
    def tfidf_response(self, user_input, threshold=0.3):
        """Approche 2: Recherche par TF-IDF"""
        processed_input = self.preprocess_text(user_input)
        user_vector = self.tfidf_vectorizer.transform([processed_input])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        if best_score > threshold:
            return self.knowledge_base[best_match_idx]['answer'], best_score, 'tfidf'
        return None, best_score, 'tfidf'
    
    def embedding_response(self, user_input, threshold=0.5):
        """Approche 3: Recherche par embeddings (Sentence-BERT)"""
        if not USE_EMBEDDINGS or self.sentence_model is None:
            return None, 0, 'embedding'
        
        user_embedding = self.sentence_model.encode([user_input])
        similarities = cosine_similarity(user_embedding, self.question_embeddings)[0]
        
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        if best_score > threshold:
            return self.knowledge_base[best_match_idx]['answer'], best_score, 'embedding'
        return None, best_score, 'embedding'
    
    def get_response(self, user_input):
        """Approche hybride: Combine toutes les techniques"""
        # Étape 1: Essayer les règles d'abord
        rule_response = self.rule_based_response(user_input)
        if rule_response:
            return rule_response, 'rule-based'
        
        # Étape 2: Essayer embeddings (meilleure qualité)
        emb_response, emb_score, _ = self.embedding_response(user_input)
        
        # Étape 3: Essayer TF-IDF
        tfidf_response, tfidf_score, _ = self.tfidf_response(user_input)
        
        # Choisir la meilleure réponse
        if emb_response and emb_score > 0.5:
            return emb_response, f'embedding (score: {emb_score:.2f})'
        elif tfidf_response and tfidf_score > 0.3:
            return tfidf_response, f'tfidf (score: {tfidf_score:.2f})'
        elif emb_response:
            return emb_response, f'embedding-low (score: {emb_score:.2f})'
        elif tfidf_response:
            return tfidf_response, f'tfidf-low (score: {tfidf_score:.2f})'
        else:
            return self.fallback_response(user_input), 'fallback'
    
    def fallback_response(self, user_input):
        """Réponse par défaut si aucune correspondance"""
        return ("Je ne suis pas sûr de bien comprendre votre question. "
                "Pourriez-vous la reformuler? Je peux vous renseigner sur:\n"
                "- Les sites touristiques (Médina, Carthage, Sidi Bou Saïd)\n"
                "- Les restaurants et la gastronomie tunisienne\n"
                "- Les transports et informations pratiques\n"
                "- L'histoire et la culture\n"
                "- Des suggestions d'itinéraires")
    
    def chat(self, user_input):
        """Fonction principale de dialogue"""
        response, method = self.get_response(user_input)
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'method': method
        })
        return response, method


def main():
    """Interface en ligne de commande"""
    print("=" * 60)
    print("🏛️  CHATBOT TOURISTIQUE - TUNIS  🇹🇳")
    print("=" * 60)
    print("\nInitialisation du chatbot...")
    
    bot = TunisChatbot()
    
    print("\n✅ Chatbot prêt!")
    print("\nTechniques utilisées:")
    print("  ✓ Règles (pattern matching)")
    print("  ✓ TF-IDF (similarité vectorielle)")
    if USE_EMBEDDINGS:
        print("  ✓ Embeddings (Sentence-BERT)")
    print("  ✓ Approche hybride")
    print("\nTapez 'quit' ou 'exit' pour quitter\n")
    print("-" * 60)
    
    # Message de bienvenue
    welcome, _ = bot.chat("bonjour")
    print(f"\n🤖 Bot: {welcome}\n")
    
    while True:
        user_input = input("👤 Vous: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'quitter', 'bye']:
            farewell, _ = bot.chat("au revoir")
            print(f"\n🤖 Bot: {farewell}\n")
            break
        
        response, method = bot.chat(user_input)
        print(f"\n🤖 Bot: {response}")
        print(f"   [Méthode: {method}]\n")


if __name__ == "__main__":
    main()