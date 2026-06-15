
from pathlib import Path
import pandas as pd
import numpy as np
from config import config

def load_and_clean(input_file: str = None) -> pd.DataFrame:
    
    input_file = input_file or config.CLEANED_DATASET
    path = config.PROJECT_ROOT / input_file
    
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    
    df = pd.read_csv(path)
    print(f"✓ Chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"  Communes : {df['Commune'].nunique()} | Années : {df['Annee'].min()}-{df['Annee'].max()}")
    
    # Nettoyage standard
    df['Commune'] = df['Commune'].str.strip().str.upper()
    
    # Détection doublons
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        print(f"⚠ {n_dup} doublons supprimés")
        df = df.drop_duplicates()
    
    # Vérifier la colonne cible
    if config.TARGET_COL not in df.columns:
        if 'Rendement_kg_ha' in df.columns:
            df[config.TARGET_COL] = df['Rendement_kg_ha']
        else:
            raise KeyError(f"Colonne cible '{config.TARGET_COL}' introuvable")
    
    # Valeurs manquantes
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠ {missing} valeurs manquantes détectées")
    
    return df
