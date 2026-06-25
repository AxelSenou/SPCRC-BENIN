import pandas as pd
import numpy as np

def create_all_features(df: pd.DataFrame,
                        commune_stats: dict = None,
                        global_stats:  dict = None) -> pd.DataFrame:
   
    df_new = df.copy()
    df_new = df_new.sort_values(['Commune', 'Annee']).reset_index(drop=True)
    
    eps = 1e-5  

    df_new['Stress_Hydrique_Semis']      = df_new['Max_Jours_Secs_Semis']      / (df_new['Pluie_Semis_mm']      + eps)
    df_new['Stress_Hydrique_Floraison']  = df_new['Max_Jours_Secs_Floraison']  / (df_new['Pluie_Floraison_mm']  + eps)
    df_new['Stress_Hydrique_Maturation'] = df_new['Max_Jours_Secs_Maturation'] / (df_new['Pluie_Maturation_mm'] + eps)
    df_new['Amplitude_Thermique']     = df_new['Temp_Max_Moy_C'] - df_new['Temp_Min_Moy_C']
    df_new['Index_Evapo_Thermique']   = (df_new['Temp_Max_Moy_C'] * df_new['Humidite_Relative_Perc']) / 100.0
    
    # Calcul de la moyenne sur le set disponible ou secours
    mean_temp_max = global_stats.get('mean_temp_max_historique', df_new['Temp_Max_Moy_C'].mean()) if global_stats else df_new['Temp_Max_Moy_C'].mean()


    df_new['Dev_Thermique_Nationale'] = df_new['Temp_Max_Moy_C'] - mean_temp_max
    df_new['Index_Hydro_Solaire']      = df_new['Pluie_Totale_mm']  / (df_new['Radiation_Solaire_MJ'] + eps)
    df_new['Ratio_ETP_Pluie']          = df_new['ETP_mm']           / (df_new['Pluie_Totale_mm']      + eps)
    df_new['Intensite_Bilan_Hydrique'] = df_new['Bilan_Hydrique_mm']/ (df_new['Pluie_Totale_mm']      + eps)
    df_new['Index_Hydro_Thermique']    = df_new['Pluie_Totale_mm']  / (df_new['Temp_Max_Moy_C']       + eps)
    df_new['Ratio_Pluie_Floraison_Semis'] = df_new['Pluie_Floraison_mm'] / (df_new['Pluie_Semis_mm']      + eps)
    df_new['Total_Jours_Secs_Critiques']  = df_new['Max_Jours_Secs_Semis'] + df_new['Max_Jours_Secs_Floraison']
    df_new['Ratio_Pluie_Critique_Mat']    = (df_new['Pluie_Floraison_mm'] + df_new['Pluie_Semis_mm']) / (df_new['Pluie_Maturation_mm'] + eps)
    df_new['Stress_Hydrique_Total']       = (df_new['Stress_Hydrique_Semis']
                                              + df_new['Stress_Hydrique_Floraison']
                                              + df_new['Stress_Hydrique_Maturation'])

    df_new['Dispo_Azote_Sol']        = df_new['Azote_g_kg'] * (1.0 - df_new['Argile_Perc'] / 100.0)
    df_new['Index_Qualite_Fixation'] = df_new['pH_Sol']      * (1.0 + df_new['Argile_Perc'] / 100.0)
    df_new['Azote_x_BH']             = df_new['Azote_g_kg']  * df_new['Bilan_Hydrique_mm'].clip(lower=0)
    df_new['pH_x_Humidite']          = df_new['pH_Sol']       * df_new['Humidite_Relative_Perc']
    df_new['Efficacite_Azote_pH']    = df_new['Azote_g_kg']   * df_new['pH_Sol']
    df_new['Efficience_Hydrique_Sol'] = (df_new['Pluie_Totale_mm'] * df_new['Argile_Perc']) / 100.0
    df_new['Score_Agro_Composite']    = (df_new['Dispo_Azote_Sol']  * df_new['Pluie_Floraison_mm']) / (df_new['Stress_Hydrique_Floraison'] + eps)
    df_new['Ratio_Pluie_Thermique'] = (df_new['Pluie_Floraison_mm'] + df_new['Pluie_Semis_mm']) / (df_new['Temp_Max_Moy_C'] + 1)
    return df_new
