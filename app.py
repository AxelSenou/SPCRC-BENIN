from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import deque
import pandas as pd
import io
import os
import sys

app = Flask(__name__)
CORS(app)

# Historique global des 10 dernières prédictions
historique_predictions = deque(maxlen=10)

try:
    from config import config
    from prediction_service import PredictionService
    prediction_service = PredictionService()
    print(" Service de prédiction SPCRC-Bénin v15.0 initialisé.")
except Exception as e:
    print(f" Erreur critique lors de l'initialisation du service : {str(e)}", file=sys.stderr)
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
        return jsonify({'success': False, 'error': 'Service de prédiction non disponible.'}), 500

    try:
        donnees = request.get_json()
        if not donnees:
            return jsonify({'success': False, 'error': 'Requête JSON manquante ou invalide.'}), 400

        # Requête multi-communes 
        if "communes" in donnees and isinstance(donnees["communes"], list):
            resultats_global = {}
            # Copie locale pour éviter les effets de bord sur le dictionnaire
            payload_base = {k: v for k, v in donnees.items() if k != "communes"}
            
            for commune in donnees["communes"]:
                payload_base['commune'] = commune
                payload_base['Commune'] = commune
                resultats_global[commune] = prediction_service.predict(payload_base, commune)
                
            return jsonify({
                'success': True,
                'mode': 'batch',
                'results': resultats_global
            }), 200

        #Requête unitaire
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
        return jsonify({'success': False, 'error': f"Erreur interne du backend : {str(e)}"}), 500


@app.route('/predict_batch_csv', methods=['POST'])
def predict_batch_csv():
    if not prediction_service:
        return jsonify({'success': False, 'error': 'Service de prédiction non initialisé.'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Aucun fichier détecté dans la requête.'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'success': False, 'error': 'Le nom du fichier est vide.'}), 400

    try:
        # Encodage 'utf-8-sig' robuste aux exports Excel avec BOM
        content = file.stream.read().decode("utf-8-sig")
        stream = io.StringIO(content, newline=None)
        df_input = pd.read_csv(stream)

        # Normalisation des noms de colonnes pour éviter les problèmes de casse
        df_input.columns = [col.strip() for col in df_input.columns]
        
        col_commune = 'Commune' if 'Commune' in df_input.columns else ('commune' if 'commune' in df_input.columns else None)
        if not col_commune:
            return jsonify({
                'success': False,
                'error': "Colonne 'Commune' introuvable dans le document CSV fourni."
            }), 400

        resultats, erreurs = [], 0
        
        for _, row in df_input.iterrows():
            # Remplacement des NaN par des valeurs par défaut pour éviter de casser le JSON type-hint
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
                        'commune':   commune,
                        'rendement': res['rendement'],
                        'categorie': res['categorie'],
                        'modele':    res['modele_utilise']
                    })
            except Exception:
                erreurs += 1

        return jsonify({
            'success':  True,
            'total':    len(resultats) + erreurs,
            'erreurs':  erreurs,
            'resultats': resultats
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f"Erreur lors de l'analyse du fichier CSV : {str(e)}"}), 500


@app.route('/historique', methods=['GET'])
def historique():
    return jsonify({'historique': list(historique_predictions)}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
