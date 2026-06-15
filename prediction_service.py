import unicodedata
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from config import config

STATS_COMMUNES = {
    "ABOMEY":        {"moyenne": 582.05,  "std": 169.13, "lag": 364.8},
    "ADJAOUERE":     {"moyenne": 984.09,  "std": 198.53, "lag": 863.2},
    "AGBANGNIZOUN":  {"moyenne": 729.73,  "std": 181.90, "lag": 887.3},
    "APLAHOUE":      {"moyenne": 635.20,  "std": 117.20, "lag": 728.8},
    "BANIKOARA":     {"moyenne": 1139.91, "std": 198.22, "lag": 1348.6},
    "BANTE":         {"moyenne": 995.10,  "std": 234.87, "lag": 902.1},
    "BASSILA":       {"moyenne": 1046.25, "std": 301.99, "lag": 1161.7},
    "BEMBEREKE":     {"moyenne": 902.56,  "std": 212.12, "lag": 1094.8},
    "BOHICON":       {"moyenne": 677.22,  "std": 187.53, "lag": 577.5},
    "BONOU":         {"moyenne": 857.96,  "std": 219.91, "lag": 1039.2},
    "BOUKOUMBE":     {"moyenne": 743.23,  "std": 389.88, "lag": 952.2},
    "COBLY":         {"moyenne": 772.47,  "std": 204.53, "lag": 773.7},
    "COPARGO":       {"moyenne": 1192.89, "std": 664.83, "lag": 991.2},
    "COVE":          {"moyenne": 889.31,  "std": 234.12, "lag": 1476.1},
    "DASSAZOUME":    {"moyenne": 1062.98, "std": 125.43, "lag": 1163.4},
    "DJAKOTOMEY":    {"moyenne": 584.78,  "std": 163.54, "lag": 882.0},
    "DJIDJA":        {"moyenne": 819.93,  "std": 117.14, "lag": 887.2},
    "DJOUGOU":       {"moyenne": 911.17,  "std": 250.18, "lag": 1115.4},
    "DOGBO":         {"moyenne": 735.42,  "std": 227.60, "lag": 1064.9},
    "GLAZOUE":       {"moyenne": 1178.87, "std": 246.90, "lag": 1457.2},
    "GOGOUNOU":      {"moyenne": 1053.26, "std": 133.64, "lag": 1037.7},
    "KALALE":        {"moyenne": 904.04,  "std": 210.27, "lag": 1161.5},
    "KANDI":         {"moyenne": 1024.71, "std": 158.93, "lag": 1006.4},
    "KARIMAMA":      {"moyenne": 816.35,  "std": 243.44, "lag": 1201.9},
    "KLOUEKANME":    {"moyenne": 770.98,  "std": 213.74, "lag": 811.4},
    "KOUANDE":       {"moyenne": 1016.48, "std": 338.34, "lag": 1624.6},
    "KEROU":         {"moyenne": 1069.31, "std": 148.45, "lag": 1321.2},
    "KETOU":         {"moyenne": 858.49,  "std": 215.14, "lag": 430.6},
    "LALO":          {"moyenne": 521.26,  "std": 379.23, "lag": 1039.2},
    "LOKOSSA":       {"moyenne": 908.84,  "std": 338.16, "lag": 1232.5},
    "MALANVILLE":    {"moyenne": 976.69,  "std": 230.47, "lag": 1323.3},
    "MATERI":        {"moyenne": 782.29,  "std": 168.99, "lag": 1140.6},
    "NDALI":         {"moyenne": 874.88,  "std": 331.85, "lag": 1006.6},
    "NATITINGOU":    {"moyenne": 1004.00, "std": 409.06, "lag": 1670.5},
    "NIKKI":         {"moyenne": 960.06,  "std": 254.63, "lag": 991.7},
    "OUAKE":         {"moyenne": 1065.14, "std": 621.02, "lag": 980.7},
    "OUINHI":        {"moyenne": 713.12,  "std": 120.69, "lag": 882.5},
    "OUESSE":        {"moyenne": 1083.03, "std": 437.18, "lag": 1078.1},
    "PARAKOU":       {"moyenne": 1232.47, "std": 473.75, "lag": 1493.4},
    "POBE":          {"moyenne": 801.59,  "std": 278.33, "lag": 474.4},
    "PERERE":        {"moyenne": 936.30,  "std": 272.12, "lag": 862.8},
    "PEHUNCO":       {"moyenne": 882.22,  "std": 103.28, "lag": 898.4},
    "SAVALOU":       {"moyenne": 927.07,  "std": 152.58, "lag": 989.5},
    "SAVE":          {"moyenne": 915.82,  "std": 252.58, "lag": 893.7},
    "SINENDE":       {"moyenne": 1008.56, "std": 204.62, "lag": 1350.0},
    "SEGBANA":       {"moyenne": 856.97,  "std": 149.99, "lag": 1123.0},
    "TANGUIETA":     {"moyenne": 872.68,  "std": 134.47, "lag": 988.1},
    "TCHAOUROU":     {"moyenne": 895.15,  "std": 170.43, "lag": 826.3},
    "TOUCOUNTOUNA":  {"moyenne": 977.50,  "std": 304.97, "lag": 1687.3},
    "TOVIKLIN":      {"moyenne": 690.72,  "std": 271.50, "lag": 1182.3},
    "ZAKPOTA":       {"moyenne": 713.44,  "std": 171.45, "lag": 886.0},
    "ZAGNANADO":     {"moyenne": 935.74,  "std": 370.15, "lag": 812.7},
    "ZOGBODOMEY":    {"moyenne": 916.66,  "std": 126.07, "lag": 935.9},
}

# Moyenne globale de fallback pour les communes totalement inconnues
_GLOBAL_MEAN = sum(v["moyenne"] for v in STATS_COMMUNES.values()) / len(STATS_COMMUNES)
_GLOBAL_STD  = sum(v["std"]     for v in STATS_COMMUNES.values()) / len(STATS_COMMUNES)
_GLOBAL_LAG  = sum(v["lag"]     for v in STATS_COMMUNES.values()) / len(STATS_COMMUNES)


class PredictionService:

    MODELE_TEMPOREL   = "meilleur_modele_temporel.json"
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

        features_temp_path = base / self.FEATURES_TEMPOREL
        if features_temp_path.exists():
            with open(features_temp_path, 'r', encoding='utf-8') as f:
                self.features_temporel = f.read().strip().split(',')
        else:
            self.features_temporel = self.features_spatial

        communes_path = base / "communes_connues.txt"
        if communes_path.exists():
            with open(communes_path, 'r', encoding='utf-8') as f:
                self.communes_connues = set(f.read().strip().split('\n'))
        else:
            self.communes_connues = {
                "ABOMEY", "ADJA-OUÈRÈ", "AGBANGNIZOUN", "APLAHOUÉ",
                "BANIKOARA", "BANTÈ", "BASSILA", "BEMBÈRÈKÈ",
                "BOHICON", "BONOU", "BOUKOUMBÉ", "COBLY",
                "COPARGO", "COVÈ", "DASSA-ZOUMÈ", "DJAKOTOMEY",
                "DJIDJA", "DJOUGOU", "DOGBO", "GLAZOUÉ",
                "GOGOUNOU", "KALALÉ", "KANDI", "KARIMAMA",
                "KLOUÉKANMÈ", "KOUANDÉ", "KÉROU", "KÉTOU",
                "LALO", "LOKOSSA", "MALANVILLE", "MATÉRI",
                "N'DALI", "NATITINGOU", "NIKKI", "OUAKÉ",
                "OUINHI", "OUÈSSE", "PARAKOU", "POBÉ",
                "PÈRÈRÈ", "PÉHUNCO", "SAVALOU", "SAVÈ",
                "SINENDÉ", "SÉGBANA", "TANGUIÉTA", "TCHAOUROU",
                "TOUCOUNTOUNA", "TOVIKLIN", "ZA-KPOTA", "ZAGNANADO",
                "ZOGBODOMEY",
            }

        print(f"✓ Service SPCRC chargé — Dual-modèle opérationnel :")
        print(f"  • Modèle spatial  : {len(self.features_spatial)} features "
              f"| {len(self.communes_connues)} communes connues")
        print(f"  • Modèle temporel : {len(self.features_temporel)} features "
              f"| fallback universel communes inconnues")

   
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

    def _get_stats(self, commune: str) -> dict:
        cle = self._normaliser(commune)
        if cle in STATS_COMMUNES:
            return STATS_COMMUNES[cle]
        # Fallback : moyenne globale
        return {"moyenne": _GLOBAL_MEAN, "std": _GLOBAL_STD, "lag": _GLOBAL_LAG}

    @staticmethod
    def _categorize(rendement: float) -> str:
        if rendement >= 1200.0:
            return "EXCELLENT"
        elif rendement >= 950.0:
            return "BON / MOYEN"
        return "FAIBLE"

    # Feature Engineering 
    def _build_inference_row(self, data_raw: dict, commune: str) -> pd.DataFrame:
        stats = self._get_stats(commune)

        p_semis = float(data_raw.get('Pluie_Semis_mm', 352.4))
        p_flor  = float(data_raw.get('Pluie_Floraison_mm', 573.7))
        p_mat   = float(data_raw.get('Pluie_Maturation_mm', 161.5))
        p_tot   = p_semis + p_flor + p_mat
        etp     = float(data_raw.get('ETP_mm', 1100.0))
        bh      = p_tot - etp

        j_semis = int(data_raw.get('Max_Jours_Secs_Semis', 2))
        j_flor  = int(data_raw.get('Max_Jours_Secs_Floraison', 1))
        j_mat   = int(data_raw.get('Max_Jours_Secs_Maturation', 12))

        t_max   = float(data_raw.get('Temp_Max_Moy_C', 33.5))
        t_min   = float(data_raw.get('Temp_Min_Moy_C', 22.0))
        hum     = float(data_raw.get('Humidite_Relative_Perc', 84.0))
        rad     = float(data_raw.get('Radiation_Solaire_MJ', 17.5))
        ph      = float(data_raw.get('pH_Sol', 6.2))
        argile  = float(data_raw.get('Argile_Perc', 22.5))
        azote   = float(data_raw.get('Azote_g_kg', 1.25))

        row = {
            # Données brutes
            'Annee':                        int(data_raw.get('annee', 2026)),
            'pH_Sol':                       ph,
            'Argile_Perc':                  argile,
            'Azote_g_kg':                   azote,
            'Pluie_Semis_mm':               p_semis,
            'Max_Jours_Secs_Semis':         j_semis,
            'Pluie_Floraison_mm':           p_flor,
            'Max_Jours_Secs_Floraison':     j_flor,
            'Pluie_Maturation_mm':          p_mat,
            'Max_Jours_Secs_Maturation':    j_mat,
            'Pluie_Totale_mm':              p_tot,
            'Temp_Max_Moy_C':               t_max,
            'Temp_Min_Moy_C':               t_min,
            'Radiation_Solaire_MJ':         rad,
            'Humidite_Relative_Perc':       hum,
            'ETP_mm':                       etp,
            'Bilan_Hydrique_mm':            bh,

            # 1. Indices de stress hydrique & thermique
            'Stress_Hydrique_Semis':        j_semis / (p_semis + 1),
            'Stress_Hydrique_Floraison':    j_flor  / (p_flor  + 1),
            'Stress_Hydrique_Maturation':   j_mat   / (p_mat   + 1),
            'Amplitude_Thermique':          t_max - t_min,
            'Index_Evapo_Thermique':        (t_max * hum) / 100,

            # 2. Interactions climatiques avancées
            'Index_Hydro_Solaire':          p_tot / (rad + 1),
            'Ratio_ETP_Pluie':              etp   / (p_tot + 1),
            'Intensite_Bilan_Hydrique':     bh    / (p_tot + 1),

            # 3. Dynamique phénologique
            'Ratio_Pluie_Floraison_Semis':  p_flor / (p_semis + 1),
            'Total_Jours_Secs_Critiques':   j_semis + j_flor,

            # 4. Sol
            'Dispo_Azote_Sol':              azote * (1 - (argile / 100)),
            'Index_Qualite_Fixation':       ph * (1 + (argile / 100)),

            # 5. Target Encoding 
            'Commune_Rendement_Moyen':      stats["moyenne"],
            'Commune_Rendement_Std':        stats["std"],
            'Rendement_Prec':               stats["lag"],
            'Dev_Thermique_Nationale':      0.0,   
        }

        return pd.DataFrame([row])

    def _preparer_matrice(self, df: pd.DataFrame,
                          colonnes: list, scaler) -> np.ndarray:
        for col in colonnes:
            if col not in df.columns:
                df[col] = 0.0
        return scaler.transform(df[colonnes].values)

  
    def predict(self, data_raw: dict, commune: str) -> dict:
        
        try:
            commune_clean = str(commune).upper().strip()
            connue = self._est_commune_connue(commune_clean)

            df_row = self._build_inference_row(data_raw, commune_clean)

            if connue:
                X = self._preparer_matrice(df_row, self.features_spatial, self.scaler_spatial)
                score = self.model_spatial.predict(X)[0]
                modele_utilise = "spatial"
            else:
                X = self._preparer_matrice(df_row, self.features_temporel, self.scaler_temporel)
                score = self.model_temporel.predict(X)[0]
                modele_utilise = "temporel"

            rendement = float(max(0.0, score))

            resultat = {
                'success':        True,
                'rendement':      round(rendement, 2),
                'categorie':      self._categorize(rendement),
                'modele_utilise': modele_utilise,
            }

            if not connue:
                resultat['avertissement'] = (
                    f"Commune '{commune}' hors périmètre d'entraînement spatial. "
                    f"Prédiction effectuée par le modèle temporel (généraliste)."
                )

            return resultat

        except Exception as e:
            return {'success': False, 'error': f"Erreur lors du calcul : {str(e)}"}

    def predict_batch(self, payload: dict) -> dict:
        return {commune: self.predict(variables, commune)
                for commune, variables in payload.items()}