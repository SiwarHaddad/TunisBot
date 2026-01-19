"""
Application Flask pour le Chatbot Touristique Tunis
Connecte le backend Python avec l'interface web
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json

# Importer notre chatbot
from tunis_chatbot import TunisChatbot

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Initialiser le chatbot
print("Initialisation du chatbot...")
bot = TunisChatbot()
print("Chatbot prêt!")

@app.route('/')
def home():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint API pour le chat"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message vide'
            }), 400
        
        # Obtenir la réponse du chatbot
        response, method = bot.chat(user_message)
        
        return jsonify({
            'success': True,
            'response': response,
            'method': method,
            'timestamp': None
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Obtenir l'historique de conversation"""
    return jsonify({
        'success': True,
        'history': bot.conversation_history
    })

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Réinitialiser la conversation"""
    bot.conversation_history = []
    return jsonify({
        'success': True,
        'message': 'Conversation réinitialisée'
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Statistiques sur les méthodes utilisées"""
    methods_count = {}
    for conv in bot.conversation_history:
        method = conv.get('method', 'unknown')
        methods_count[method] = methods_count.get(method, 0) + 1
    
    return jsonify({
        'success': True,
        'total_messages': len(bot.conversation_history),
        'methods_distribution': methods_count
    })

if __name__ == '__main__':
    print("\n" +
          "="*60)
    print("🏛️  CHATBOT TOURISTIQUE TUNIS - SERVEUR WEB")
    print("="*60)
    print("\nServeur démarré sur: http://localhost:5000")
    print("\nTechniques utilisées:")
    print("  ✓ Règles (pattern matching)")
    print("  ✓ TF-IDF (similarité vectorielle)")
    print("  ✓ Embeddings (Sentence-BERT)")
    print("  ✓ Approche hybride")
    print("\nAppuyez sur Ctrl+C pour arrêter le serveur")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)