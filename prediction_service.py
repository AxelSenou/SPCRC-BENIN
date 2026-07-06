# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import numpy as np
from config import config
from feature_engineering import create_all_features

class PredictionService:
    def __init__(self):
        base = config.OUTPUT_DIR

        # 1. Chargement Artefacts Spatiaux
        self.model_spatial  = joblib.load(base / "meilleur_modele_spatial.joblib")
        self.scaler_spatial = joblib.load(base / "scaler_spatial.joblib")
        self.pca_spatial    = joblib.load(base / "pca_spatial.joblib") 
        with open(base / "features_spatial.txt", 'r', encoding='utf-8') as f:
            self.features_spatial = f.read().strip().split(',')

        # 2. Chargement Artefacts Temporels
        self.model_temporel  = joblib.load(base / "meilleur_modele_temporel.joblib")
        self.scaler_temporel = joblib.load(base / "scaler_temporel.joblib")
        self.pca_temporel    = joblib.load(base / "pca_temporel.joblib") 
        with open(base / "features_temporel.txt", 'r', encoding='utf-8') as f:
            self.features_temporel = f.read().strip().split(',')

        # Liste des communes connues 
        with open(base / "communes_connues.txt", 'r', encoding='utf-8') as f:
            self.communes_visitees = set(line.strip().upper() for line in f)

    def _preparer_matrice(self, df_row, features_attendues, scaler_utilise, pca_utilise):
       
        # Feature Engineering 
        df_engineered = create_all_features(df_row)
        df_final = df_engineered[features_attendues]
        
        # Transformation Standard
        X_scaled = scaler_utilise.transform(df_final)
        
        # Projection PCA 
        X_pca = pca_utilise.transform(X_scaled)
        return X_pca

    def predict(self, data_dict, commune_nom):
        commune = str(commune_nom).upper().strip()
        connue = commune in self.communes_visitees

        # Création d'un DataFrame à une ligne à partir du dictionnaire web
        df_row = pd.DataFrame([data_dict])
        if 'Commune' not in df_row.columns:
            df_row['Commune'] = commune
        if 'Annee' not in df_row.columns:
            df_row['Annee'] = 2026

        try:
            if connue:
                # Modèle Spatial + PCA Spatial
                X = self._preparer_matrice(df_row, self.features_spatial, self.scaler_spatial, self.pca_spatial)
                score = self.model_spatial.predict(X)[0]
                modele_utilise = "spatial"
            else:
                # Modèle Temporel + PCA Temporel
                X = self._preparer_matrice(df_row, self.features_temporel, self.scaler_temporel, self.pca_temporel)
                score = self.model_temporel.predict(X)[0]
                modele_utilise = "temporel"

            rendement = float(max(0.0, score))
            
            resultat = {
                'success': True,
                'rendement': round(rendement, 2),
                'categorie': "Bon" if rendement > 1000 else "Moyen", # Remplacez par votre fonction de catégorisation originale
                'modele_utilise': modele_utilise
            }
            if not connue:
                resultat['avertissement'] = f"Commune '{commune}' hors périmètre spatial. Utilisation du modèle temporel."
            
            return resultat
        except Exception as e:
            return {'success': False, 'error': str(e)}