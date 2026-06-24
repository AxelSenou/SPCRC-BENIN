# -*- coding: utf-8 -*-
"""
SPCRC-Bénin — Service de Prédiction v15.0
Features purement pédoclimatiques — dual-modèle spatial/temporel
"""
import json
import joblib
import unicodedata
import numpy as np
import pandas as pd
from config import config


class PredictionService:

    # Noms des artefacts temporels
    MODELE_TEMPOREL   = "meilleur_modele_temporel.joblib"
    SCALER_TEMPOREL   = "scaler_temporel.joblib"
    FEATURES_TEMPOREL = "features_temporel.txt"

    def __init__(self):
        base = config.OUTPUT_DIR

        # ── Modèle SPATIAL ────────────────────────────────────────────────────
        self.model_spatial  = joblib.load(base / config.MODEL_PATH)
        self.scaler_spatial = joblib.load(base / config.SCALER_PATH)
        with open(base / config.FEATURES_PATH, 'r', encoding='utf-8') as f:
            self.features_spatial = f.read().strip().split(',')

        # ── Modèle TEMPOREL ───────────────────────────────────────────────────
        self.model_temporel  = joblib.load(base / self.MODELE_TEMPOREL)
        self.scaler_temporel = joblib.load(base / self.SCALER_TEMPOREL)
        with open(base / self.FEATURES_TEMPOREL, 'r', encoding='utf-8') as f:
            self.features_temporel = f.read().strip().split(',')

        # ── Communes connues ──────────────────────────────────────────────────
        communes_path = base / "communes_connues.txt"
        if communes_path.exists():
            with open(communes_path, 'r', encoding='utf-8') as f:
                self.communes_connues = set(
                    line.strip() for line in f if line.strip()
                )
        else:
            self.communes_connues = set()

        # ── Statistiques globales (mean_temp_max pour Dev_Thermique) ─────────
        stats_path = base / "statistiques_globales.json"
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.global_stats = json.load(f)
        else:
            self.global_stats = {'mean_temp_max_historique': 33.5}

        self.mean_temp_max = self.global_stats.get('mean_temp_max_historique', 33.5)

        print(f"✓ Service SPCRC chargé — Dual-modèle pédoclimatique :")
        print(f"  • Modèle spatial  : {len(self.features_spatial)} features "
              f"| {len(self.communes_connues)} communes connues")
        print(f"  • Modèle temporel : {len(self.features_temporel)} features "
              f"| fallback universel")
        print(f"  • Temp max nationale : {self.mean_temp_max:.2f}°C")

    # ── Utilitaires ───────────────────────────────────────────────────────────
    @staticmethod
    def _normaliser(nom: str) -> str:
        nfkd = unicodedata.normalize('NFD', str(nom))
        return (nfkd.encode('ascii', 'ignore').decode('utf-8')
                .upper().strip()
                .replace("'", "").replace("'", "")
                .replace("-", "").replace(" ", ""))

    def _est_commune_connue(self, commune: str) -> bool:
        if commune in self.communes_connues:
            return True
        norm = self._normaliser(commune)
        return any(self._normaliser(c) == norm for c in self.communes_connues)

    @staticmethod
    def _categorize(rendement: float) -> str:
        if rendement >= 1200.0:
            return "EXCELLENT"
        elif rendement >= 950.0:
            return "BON / MOYEN"
        return "FAIBLE"

    # ── Feature Engineering (miroir de feature_engineering.py) ───────────────
    def _build_features(self, data_raw: dict) -> pd.DataFrame:
        p_semis = float(data_raw.get('Pluie_Semis_mm',      352.4))
        p_flor  = float(data_raw.get('Pluie_Floraison_mm',  573.7))
        p_mat   = float(data_raw.get('Pluie_Maturation_mm', 161.5))
        p_tot   = float(data_raw.get('Pluie_Totale_mm', p_semis + p_flor + p_mat))
        etp     = float(data_raw.get('ETP_mm',      1100.0))
        bh      = float(data_raw.get('Bilan_Hydrique_mm', p_tot - etp))
        j_semis = float(data_raw.get('Max_Jours_Secs_Semis',      2))
        j_flor  = float(data_raw.get('Max_Jours_Secs_Floraison',  1))
        j_mat   = float(data_raw.get('Max_Jours_Secs_Maturation', 12))
        t_max   = float(data_raw.get('Temp_Max_Moy_C',         33.5))
        t_min   = float(data_raw.get('Temp_Min_Moy_C',         22.0))
        hum     = float(data_raw.get('Humidite_Relative_Perc', 84.0))
        rad     = float(data_raw.get('Radiation_Solaire_MJ',   17.5))
        ph      = float(data_raw.get('pH_Sol',      6.2))
        argile  = float(data_raw.get('Argile_Perc', 22.5))
        azote   = float(data_raw.get('Azote_g_kg',  1.25))

        sh_semis = j_semis / (p_semis + 1)
        sh_flor  = j_flor  / (p_flor  + 1)
        sh_mat   = j_mat   / (p_mat   + 1)
        dispo_az = azote * (1 - argile / 100)

        row = {
            # Brutes
            'Pluie_Semis_mm':               p_semis,
            'Max_Jours_Secs_Semis':         j_semis,
            'Pluie_Floraison_mm':           p_flor,
            'Max_Jours_Secs_Floraison':     j_flor,
            'Pluie_Maturation_mm':          p_mat,
            'Max_Jours_Secs_Maturation':    j_mat,
            'pH_Sol':                       ph,
            'Argile_Perc':                  argile,
            'Azote_g_kg':                   azote,
            'Pluie_Totale_mm':              p_tot,
            'Temp_Max_Moy_C':               t_max,
            'Temp_Min_Moy_C':               t_min,
            'Radiation_Solaire_MJ':         rad,
            'Humidite_Relative_Perc':       hum,
            'ETP_mm':                       etp,
            'Bilan_Hydrique_mm':            bh,
            # Stress hydrique
            'Stress_Hydrique_Semis':        sh_semis,
            'Stress_Hydrique_Floraison':    sh_flor,
            'Stress_Hydrique_Maturation':   sh_mat,
            'Stress_Hydrique_Total':        sh_semis + sh_flor + sh_mat,
            # Thermique
            'Amplitude_Thermique':          t_max - t_min,
            'Index_Evapo_Thermique':        (t_max * hum) / 100,
            'Dev_Thermique_Nationale':      t_max - self.mean_temp_max,
            # Interactions climatiques
            'Index_Hydro_Solaire':          p_tot / (rad + 1),
            'Ratio_ETP_Pluie':              etp   / (p_tot + 1),
            'Intensite_Bilan_Hydrique':     bh    / (p_tot + 1),
            'Index_Hydro_Thermique':        p_tot / (t_max + 1),
            # Phénologique
            'Ratio_Pluie_Floraison_Semis':  p_flor / (p_semis + 1),
            'Total_Jours_Secs_Critiques':   j_semis + j_flor,
            'Ratio_Pluie_Critique_Mat':     (p_flor + p_semis) / (p_mat + 1),
            # Sol
            'Dispo_Azote_Sol':              dispo_az,
            'Index_Qualite_Fixation':       ph * (1 + argile / 100),
            'Azote_x_BH':                   azote * max(0.0, bh),
            'pH_x_Humidite':                ph * hum,
            'Efficacite_Azote_pH':          azote * ph,
            # Sol × Climat
            'Efficience_Hydrique_Sol':      (p_tot * argile) / 100,
            'Score_Agro_Composite':         (dispo_az * p_flor) / (sh_flor + 1),
        }

        return pd.DataFrame([row])

    def _preparer_matrice(self, df: pd.DataFrame,
                          features: list, scaler) -> np.ndarray:
        for col in features:
            if col not in df.columns:
                df[col] = 0.0
        return scaler.transform(df[features].values)

    # ── Prédiction unitaire (endpoint /predict JSON) ──────────────────────────
    def predict(self, data_raw: dict, commune: str) -> dict:
        try:
            commune_clean = str(commune).upper().strip()
            connue = self._est_commune_connue(commune_clean)

            df_row = self._build_features(data_raw)

            if connue:
                X = self._preparer_matrice(
                    df_row, self.features_spatial, self.scaler_spatial)
                score = self.model_spatial.predict(X)[0]
                modele_utilise = "spatial"
            else:
                X = self._preparer_matrice(
                    df_row, self.features_temporel, self.scaler_temporel)
                score = self.model_temporel.predict(X)[0]
                modele_utilise = "temporel"

            rendement = float(max(0.0, score))
            resultat  = {
                'success':        True,
                'rendement':      round(rendement, 2),
                'categorie':      self._categorize(rendement),
                'modele_utilise': modele_utilise,
            }
            if not connue:
                resultat['avertissement'] = (
                    f"Commune '{commune}' hors périmètre spatial. "
                    f"Prédiction par le modèle temporel (généraliste)."
                )
            return resultat

        except Exception as e:
            return {'success': False, 'error': f"Erreur : {str(e)}"}

    # ── Compatibilité app.py (predict_unitaire) ───────────────────────────────
    def predict_unitaire(self, data_dict: dict) -> float:
        commune = str(data_dict.get('Commune', '')).upper().strip()
        res = self.predict(data_dict, commune)
        if not res['success']:
            raise ValueError(res['error'])
        return res['rendement']

    # ── Prédiction batch ──────────────────────────────────────────────────────
    def predict_batch(self, payload: dict) -> dict:
        return {commune: self.predict(variables, commune)
                for commune, variables in payload.items()}