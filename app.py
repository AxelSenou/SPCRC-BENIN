from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from config import config
from prediction_service import PredictionService

app = Flask(__name__)
CORS(app)

try:
    prediction_service = PredictionService()
    print("✓ Le service de prédiction SPCRC est opérationnel pour la soutenance.")
except Exception as e:
    print(f" Erreur critique lors de l'initialisation du service : {e}")
    prediction_service = None


@app.route('/')
def home():
    return ( 'generer_carte.html')


@app.route('/predict', methods=['POST'])
def predict():
   
    if not prediction_service:
        return jsonify({'success': False, 'error': 'Service non initialisé.'}), 500

    try:
        donnees = request.get_json()
        if not donnees:
            return jsonify({'success': False, 'error': 'Requête JSON manquante.'}), 400

        # Requête multi-communes
        if "communes" in donnees and isinstance(donnees["communes"], list):
            resultats_global = {}
            for commune in donnees["communes"]:
                resultats_global[commune] = prediction_service.predict(donnees, commune)
            return jsonify({'success': True, 'mode': 'batch', 'results': resultats_global}), 200

        #  Requête  unitaire
        commune_brute = donnees.get('commune')
        if not commune_brute:
            return jsonify({'success': False, 'error': 'Le champ commune est requis.'}), 400

        res_unitaire = prediction_service.predict(donnees, commune_brute)
        return jsonify(res_unitaire), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f"Erreur backend : {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'active', 'model_loaded': prediction_service is not None}), 200



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)