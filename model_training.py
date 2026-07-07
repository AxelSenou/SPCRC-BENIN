import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

from config import config
from data_loader import load_and_clean
from feature_engineering import create_all_features

warnings.filterwarnings('ignore')

SEP  = "=" * 70
print(f"\n{SEP}")
print("        SPCRC-BÉNIN · PIPELINE TEMPOREL")
print(SEP)

# Chargement et Feature Engineering
df_brut = load_and_clean()
y_brut  = df_brut['Rendement_kg_ha'].values
bins_y  = pd.qcut(y_brut, q=4, labels=False, duplicates='drop')

print(f"\n Dataset : {df_brut.shape[0]} lignes | {df_brut['Commune'].nunique()} communes")

df_engineered = create_all_features(df_brut)
X_raw = df_engineered.drop(columns=['Commune', 'Annee', 'Rendement_kg_ha'], errors='ignore')
y = df_engineered['Rendement_kg_ha'].values
features_names = X_raw.columns.tolist()

# Split Temporel 
df_train_raw, df_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=bins_y
)
print(f"    Split stratifié : {df_train_raw.shape[0]} train | {df_test_raw.shape[0]} test")

# Normalisation et PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train_raw)
X_test_scaled = scaler.transform(df_test_raw)

pca = PCA(n_components=12, random_state=config.RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

variance_expliquee = np.sum(pca.explained_variance_ratio_) * 100
print(f"    PCA : 38 features condensées en 12 composantes (Variance expliquée : {variance_expliquee:.2f}%)")

# Modèles
modeles = {
    "RandomForest_Temp": RandomForestRegressor(
        n_estimators=150, max_depth=6, min_samples_leaf=5,
        oob_score=True, random_state=config.RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost_Temp": xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=5,
        reg_lambda=10,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
}

# Entraînement et Validation
meilleurs_scores = {}
meilleurs_modeles = {}

print(f"\n{'-'*70}\n ENTRAÎNEMENT & EVALUATION\n{'-'*70}")
kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

for nom, model in modeles.items():
    model.fit(X_train_pca, y_train)
    preds_test = model.predict(X_test_pca)
    preds_train = model.predict(X_train_pca)
    
    r2_train = r2_score(y_train, preds_train)
    r2_test = r2_score(y_test, preds_test)
    mae_test = mean_absolute_error(y_test, preds_test)
    
    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=kf, scoring='r2')
    
    print(f"\n  {nom}")
    print(f"     R² Train  : {r2_train:.4f}")
    print(f"     R² Test   : {r2_test:.4f}  (Écart={abs(r2_train - r2_test):.4f})")
    print(f"     CV R²     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"     MAE Test  : {mae_test:.2f} kg/ha")
    
    meilleurs_scores[nom] = r2_test
    meilleurs_modeles[nom] = model

# Gagnant et Sauvegarde
nom_gagnant = max(meilleurs_scores, key=meilleurs_scores.get)
modele_gagnant = meilleurs_modeles[nom_gagnant]

print(f"\n{SEP}\n RAPPORT FINAL — PIPELINE TEMPOREL\n{SEP}")
print(f"  Modèle Champion : {nom_gagnant}")
print(f"  R² Test         : {meilleurs_scores[nom_gagnant]:.4f}")

joblib.dump(modele_gagnant, config.OUTPUT_DIR / "meilleur_modele_temporel.joblib")
joblib.dump(scaler, config.OUTPUT_DIR / "scaler_temporel.joblib")
joblib.dump(pca, config.OUTPUT_DIR / "pca_temporel.joblib")

with open(config.OUTPUT_DIR / "features_temporel.txt", "w", encoding="utf-8") as f:
    f.write(",".join(features_names))

print(f"\n Artefacts et modèle temporel sauvegardés dans {config.OUTPUT_DIR}")