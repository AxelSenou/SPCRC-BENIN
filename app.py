
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import deque
import pandas as pd
import io
import os
import sys
from config import config
from prediction_service import PredictionService

app = Flask(__name__)
CORS(app)

# Historique global en mémoire des 10 dernières simulations
historique_predictions = deque(maxlen=10)
 
prediction_service = PredictionService()
print(" Service de prédiction SPCRC-Bénin initialisé.")



@app.route('/')
def home():
    return send_from_directory('.', 'generer_carte.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'active',
        'model_loaded': prediction_service is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    if not prediction_service:
        return jsonify({'success': False, 'error': 'Le service de prédiction est indisponible.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Aucune donnée JSON reçue.'}), 400

    commune = str(data.get('Commune', '')).upper().strip()
    if not commune:
        return jsonify({'success': False, 'error': "Le champ 'Commune' est obligatoire."}), 400

    try:
        # Appel du service qui applique create_all_features -> Scaler -> PCA -> Modèle Gagnant
        res = prediction_service.predict(data, commune)
        
        if res.get('success'):
            # Enregistrement dans l'historique de l'API
            historique_predictions.appendleft({
                'commune': commune,
                'rendement': res['rendement'],
                'categorie': res['categorie'],
                'modele': res['modele_utilise']
            })
        return jsonify(res), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not prediction_service:
        return jsonify({'success': False, 'error': 'Le service de prédiction est indisponible.'}), 503

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': "Le formulaire doit contenir un fichier sous la clé 'file'."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Fichier non sélectionné.'}), 400

    try:
        # Lecture du flux de données CSV en mémoire avec Pandas
        stream = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
        df = pd.read_csv(stream)

        col_commune = None
        for col in df.columns:
            if col.lower().strip() == 'commune':
                col_commune = col
                break

        if not col_commune:
            return jsonify({'success': False, 'error': "Le fichier CSV doit contenir une colonne nommée 'Commune'."}), 400

        resultats = []
        erreurs = 0

        for _, row in df.iterrows():
            d = row.dropna().to_dict()
            commune = str(d.get(col_commune, '')).upper().strip()
            
            if not commune:
                erreurs += 1
                continue
                
            try:
                res = prediction_service.predict(d, commune)
                res['commune'] = commune
                resultats.append(res)
                
                if res.get('success'):
                    historique_predictions.appendleft({
                        'commune': commune,
                        'rendement': res['rendement'],
                        'categorie': res['categorie'],
                        'modele': res['modele_utilise']
                    })
            except Exception:
                erreurs += 1

        return jsonify({
            'success': True,
            'total': len(resultats) + erreurs,
            'erreurs': erreurs,
            'resultats': resultats
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f"Erreur lors de l'analyse du fichier CSV : {str(e)}"}), 500


@app.route('/historique', methods=['GET'])
def historique():
    return jsonify({'historique': list(historique_predictions)}), 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)