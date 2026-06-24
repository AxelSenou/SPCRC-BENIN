# -*- coding: utf-8 -*-
"""
SPCRC-Bénin — Serveur Flask v15.0
Prédiction unitaire et batch (cartographie globale)
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import deque
import pandas as pd
import io
import os

from config import config
from prediction_service import PredictionService

app = Flask(__name__)
CORS(app)

historique_predictions = deque(maxlen=10)

try:
    prediction_service = PredictionService()
    print("✓ Service de prédiction SPCRC-Bénin v15.0 initialisé.")
except Exception as e:
    print(f"❌ Erreur critique : {str(e)}")
    prediction_service = None


@app.route('/')
def home():
    return send_from_directory('.', 'generer_carte.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'active',
        'model_loaded': prediction_service is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    if not prediction_service:
        return jsonify({'success': False, 'error': 'Service non initialisé.'}), 500

    try:
        donnees = request.get_json()
        if not donnees:
            return jsonify({'success': False, 'error': 'Requête JSON manquante.'}), 400

        # ── Requête multi-communes (cartographie globale) ─────────────────────
        if "communes" in donnees and isinstance(donnees["communes"], list):
            resultats_global = {}
            for commune in donnees["communes"]:
                resultats_global[commune] = prediction_service.predict(donnees, commune)
            return jsonify({
                'success': True,
                'mode': 'batch',
                'results': resultats_global
            }), 200

        # ── Requête unitaire ──────────────────────────────────────────────────
        commune_brute = donnees.get('commune') or donnees.get('Commune')
        if not commune_brute:
            return jsonify({'success': False, 'error': 'Le champ commune est requis.'}), 400

        res = prediction_service.predict(donnees, commune_brute)

        if res.get('success'):
            historique_predictions.appendleft({
                'commune': str(commune_brute).upper(),
                'rendement': res['rendement'],
                'categorie': res['categorie'],
                'modele': res['modele_utilise']
            })

        return jsonify(res), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f"Erreur backend : {str(e)}"}), 500


@app.route('/predict_batch_csv', methods=['POST'])
def predict_batch_csv():
    """Upload CSV et prédiction ligne par ligne"""
    if not prediction_service:
        return jsonify({'success': False, 'error': 'Service non initialisé.'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Aucun fichier détecté.'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'success': False, 'error': 'Nom de fichier vide.'}), 400

    try:
        stream   = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
        df_input = pd.read_csv(stream)

        if 'Commune' not in df_input.columns:
            return jsonify({
                'success': False,
                'error': "Colonne 'Commune' introuvable dans le CSV."
            }), 400

        resultats, erreurs = [], 0
        for _, row in df_input.iterrows():
            d = row.to_dict()
            commune = str(d.get('Commune', '')).upper().strip()
            try:
                res = prediction_service.predict(d, commune)
                res['commune'] = commune
                resultats.append(res)
                if res.get('success'):
                    historique_predictions.appendleft({
                        'commune':   commune,
                        'rendement': res['rendement'],
                        'categorie': res['categorie'],
                        'modele':    res['modele_utilise']
                    })
            except Exception:
                erreurs += 1

        return jsonify({
            'success':  True,
            'total':    len(resultats),
            'erreurs':  erreurs,
            'resultats': resultats
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/historique', methods=['GET'])
def historique():
    return jsonify({'historique': list(historique_predictions)}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)