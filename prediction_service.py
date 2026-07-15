import joblib
import pandas as pd
from config import config
from feature_engineering import create_all_features

class PredictionService:
    def __init__(self):
        base = config.OUTPUT_DIR

        # Chargement de l'artefact temporel
        self.model_temporel  = joblib.load(base / "meilleur_modele_temporel.joblib")
        self.scaler_temporel = joblib.load(base / "scaler_temporel.joblib")
        self.pca_temporel    = joblib.load(base / "pca_temporel.joblib")
        with open(base / "features_temporel.txt", 'r', encoding='utf-8') as f:
            self.features_temporel = f.read().strip().split(',')

        

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

        normalized = {}
        for key, value in (data_dict or {}).items():
            if value is None:
                continue
            normalized[str(key)] = value

        # Création d'un DataFrame à une ligne à partir du dictionnaire web
        df_row = pd.DataFrame([normalized])
        if 'Commune' not in df_row.columns:
            df_row['Commune'] = commune
        if 'Annee' not in df_row.columns:
            df_row['Annee'] = 2026

        # Valeurs par défaut si certaines variables essentielles sont absentes
        defaults = {
            'Pluie_Semis_mm': 250.0,
            'Pluie_Floraison_mm': 320.0,
            'Pluie_Maturation_mm': 280.0,
            'Pluie_Totale_mm': 850.0,
            'Temp_Max_Moy_C': 33.0,
            'Temp_Min_Moy_C': 24.0,
            'Humidite_Relative_Perc': 70.0,
            'Radiation_Solaire_MJ': 18.0,
            'ETP_mm': 110.0,
            'Bilan_Hydrique_mm': 40.0,
            'Max_Jours_Secs_Semis': 6.0,
            'Max_Jours_Secs_Floraison': 4.0,
            'Max_Jours_Secs_Maturation': 5.0,
            'pH_Sol': 6.2,
            'Argile_Perc': 18.0,
            'Azote_g_kg': 1.2,
        }
        for key, value in defaults.items():
            if key not in df_row.columns:
                df_row[key] = value
            elif pd.isna(df_row.loc[0, key]):
                df_row.loc[0, key] = value

        try:
            # Modèle Temporel + PCA Temporel
            X = self._preparer_matrice(df_row, self.features_temporel, self.scaler_temporel, self.pca_temporel)
            score = self.model_temporel.predict(X)[0]
            modele_utilise = "temporel"

            rendement = float(max(0.0, score))
            
            resultat = {
                'success': True,
                'rendement': round(rendement, 2),
                'categorie': "Bon" if rendement > 1000 else "Moyen",
                'modele_utilise': modele_utilise
            }
            return resultat
        except Exception as e:
            return {'success': False, 'error': str(e)}