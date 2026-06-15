import pandas as pd
import numpy as np

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:

    df_new = df.copy()
    
    # --- Tri indispensable pour les calculs de Lags ---
    df_new = df_new.sort_values(by=['Commune', 'Annee']).reset_index(drop=True)

    # ==========================================
    # 1. INDICES DE STRESS HYDRIQUE & THERMIQUE
    # ==========================================
    df_new['Stress_Hydrique_Semis'] = df_new['Max_Jours_Secs_Semis'] / (df_new['Pluie_Semis_mm'] + 1)
    df_new['Stress_Hydrique_Floraison'] = df_new['Max_Jours_Secs_Floraison'] / (df_new['Pluie_Floraison_mm'] + 1)
    df_new['Stress_Hydrique_Maturation'] = df_new['Max_Jours_Secs_Maturation'] / (df_new['Pluie_Maturation_mm'] + 1)
    
    # Amplitude thermique diurne
    df_new['Amplitude_Thermique'] = df_new['Temp_Max_Moy_C'] - df_new['Temp_Min_Moy_C']
    
    # Index Évapo-Thermique composite
    df_new['Index_Evapo_Thermique'] = (df_new['Temp_Max_Moy_C'] * df_new['Humidite_Relative_Perc']) / 100

    # ==========================================
    # 2. INTERACTIONS CLIMATIQUES AVANCÉES
    # ==========================================
    # Ratio d'efficience pluie / rayonnement (Photosynthèse vs Stress)
    df_new['Index_Hydro_Solaire'] = df_new['Pluie_Totale_mm'] / (df_new['Radiation_Solaire_MJ'] + 1)
    
    # Demande en eau vs Précipitations
    df_new['Ratio_ETP_Pluie'] = df_new['ETP_mm'] / (df_new['Pluie_Totale_mm'] + 1)
    
    # Balance hydrique relative à la pluie reçue
    df_new['Intensite_Bilan_Hydrique'] = df_new['Bilan_Hydrique_mm'] / (df_new['Pluie_Totale_mm'] + 1)

    # ==========================================
    # 3. DYNAMIQUE ET TRANSITION PHÉNOLOGIQUE
    # ==========================================
    # Ratio d'approvisionnement en eau Semis vs Floraison
    df_new['Ratio_Pluie_Floraison_Semis'] = df_new['Pluie_Floraison_mm'] / (df_new['Pluie_Semis_mm'] + 1)
    
    # Cumul de la pression sèche sur les phases de croissance active
    df_new['Total_Jours_Secs_Critiques'] = df_new['Max_Jours_Secs_Semis'] + df_new['Max_Jours_Secs_Floraison']

    # ==========================================
    # 4. CARACTÉRISTIQUES DU SOL
    # ==========================================
    # Disponibilité estimée de l'Azote (pénalisée par le taux d'argile/lessivage)
    df_new['Dispo_Azote_Sol'] = df_new['Azote_g_kg'] * (1 - (df_new['Argile_Perc'] / 100))
    
    # Capacité d'échange structurelle estimée (Interaction pH et Argile)
    df_new['Index_Qualite_Fixation'] = df_new['pH_Sol'] * (1 + (df_new['Argile_Perc'] / 100))

    # ==========================================
    # 5. VARIABLES DE ZONE ET STATISTIQUES HISTORIQUES
    # ==========================================
    # Moyenne historique du rendement par Commune (Target Encoding hors-ligne)
    commune_means = df_new.groupby('Commune')['Rendement_kg_ha'].transform('mean')
    df_new['Commune_Rendement_Moyen'] = commune_means
    
    commune_stds = df_new.groupby('Commune')['Rendement_kg_ha'].transform('std')
    df_new['Commune_Rendement_Std'] = commune_stds.fillna(0)
    
    # Rendement de l'année précédente (Lag 1) - Très puissant pour capter la mémoire du sol
    df_new['Rendement_Prec'] = df_new.groupby('Commune')['Rendement_kg_ha'].shift(1)
    
    # Imputation de sécurité pour la première année enregistrée de chaque commune
    df_new['Rendement_Prec'] = df_new['Rendement_Prec'].fillna(df_new['Commune_Rendement_Moyen'])

    # Moyenne nationale glissante thermique (Déviation micro-climatique régionale)
    mean_temp_nat = df_new['Temp_Max_Moy_C'].mean()
    df_new['Dev_Thermique_Nationale'] = df_new['Temp_Max_Moy_C'] - mean_temp_nat
    
    # Sécurité anti-valeurs manquantes résiduelles
    global_mean = df_new['Rendement_kg_ha'].mean()
    df_new['Commune_Rendement_Moyen'] = df_new['Commune_Rendement_Moyen'].fillna(global_mean)
    df_new['Rendement_Prec'] = df_new['Rendement_Prec'].fillna(global_mean)

    return df_new