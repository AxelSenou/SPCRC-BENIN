import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

from config import config
from data_loader import load_and_clean
from feature_engineering import create_all_features
from model_evaluation import compare_models

warnings.filterwarnings('ignore')

SEP  = "=" * 70
SEP2 = "-" * 70

print(f"\n{SEP}")
print("  SPCRC-BÉNIN · PIPELINE TEMPOREL v15.4 — PÉDOCLIMATIQUE PUR")
print(SEP)

# ── 1. Chargement ─────────────────────────────────────────────────────────────
df_brut = load_and_clean()
y_brut  = df_brut['Rendement_kg_ha'].values
bins_y  = pd.qcut(y_brut, q=4, labels=False, duplicates='drop')
print(f"\n  Dataset : {df_brut.shape[0]} lignes | "
      f"{df_brut['Commune'].nunique()} communes | "
      f"{df_brut['Annee'].min()}-{df_brut['Annee'].max()}")

# ── 2. Split stratifié ────────────────────────────────────────────────────────
df_train_raw, df_test_raw = train_test_split(
    df_brut, test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE, stratify=bins_y
)
print(f"\n✓ Split stratifié : {len(df_train_raw)} train | {len(df_test_raw)} test")
print(f"  Communes train : {df_train_raw['Commune'].nunique()} | "
      f"Communes test : {df_test_raw['Commune'].nunique()}")

mean_temp_max_train = float(df_train_raw['Temp_Max_Moy_C'].mean())

# ── 3. Feature Engineering ────────────────────────────────────────────────────
df_train_fe = create_all_features(df_train_raw)
df_test_fe  = create_all_features(df_test_raw)

y_train    = df_train_fe[config.TARGET_COL].values
y_test     = df_test_fe[config.TARGET_COL].values
X_train_df = df_train_fe.drop(columns=['Annee', 'Commune', config.TARGET_COL], errors='ignore')
liste_colonnes = X_train_df.columns.tolist()

print(f"\n  Features pédoclimatiques : {len(liste_colonnes)}")
print(f"  Train : {len(y_train)} obs | Test : {len(y_test)} obs")

scaler_X       = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_df.values)
X_test_scaled  = scaler_X.transform(df_test_fe[liste_colonnes].values)

# ── 4. Sélection des features (ExtraTrees — top 16) ──────────────────────────
print(f"\n{SEP2}")
print("  SÉLECTION DES FEATURES (ExtraTrees — top 16)")
print(SEP2)

selector = ExtraTreesRegressor(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1)
selector.fit(X_train_scaled, y_train)

imp            = selector.feature_importances_
top_idx        = np.argsort(imp)[::-1][:16]
top_idx_sorted = sorted(top_idx)
features_sel   = [liste_colonnes[i] for i in top_idx_sorted]
X_train_sel    = X_train_scaled[:, top_idx_sorted]
X_test_sel     = X_test_scaled[:,  top_idx_sorted]

print(f"\n  {'Feature':<40} Importance")
print(f"  {'-'*52}")
for i, idx in enumerate(np.argsort(imp)[::-1][:16]):
    print(f"  {i+1:2d}. {liste_colonnes[idx]:<37} {imp[idx]:.4f}")

# ── 5. Entraînement ───────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  ENTRAÎNEMENT")
print(SEP2)

dictionnaire_modeles = {
    "RandomForest_Temp_v15": RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        max_features='sqrt',
        oob_score=True,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    ),
    "XGBoost_Temp_v15": xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=4,
        reg_alpha=5.0,
        reg_lambda=10.0,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
}

for nom, mod in dictionnaire_modeles.items():
    print(f"\n  → {nom}")
    mod.fit(X_train_sel, y_train)
    if hasattr(mod, 'oob_score_'):
        print(f"     OOB R²    : {mod.oob_score_:.4f}")
    r2_tr  = r2_score(y_train, mod.predict(X_train_sel))
    r2_te  = r2_score(y_test,  mod.predict(X_test_sel))
    mae_te = mean_absolute_error(y_test, mod.predict(X_test_sel))
    rmse   = np.sqrt(mean_squared_error(y_test, mod.predict(X_test_sel)))
    ecart  = r2_tr - r2_te
    print(f"     R² Train  : {r2_tr:.4f}")
    print(f"     R² Test   : {r2_te:.4f}  (écart={ecart:.4f} {'✓ OK' if ecart < 0.25 else '⚠ Overfitting'})")
    print(f"     MAE Test  : {mae_te:.2f} kg/ha")
    print(f"     RMSE Test : {rmse:.2f} kg/ha")

# ── 6. Validation croisée ─────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  VALIDATION CROISÉE 5-FOLD")
print(SEP2)
kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
for nom, mod in dictionnaire_modeles.items():
    cv_r2  = cross_val_score(mod, X_train_sel, y_train, cv=kf, scoring='r2', n_jobs=-1)
    cv_mae = cross_val_score(mod, X_train_sel, y_train, cv=kf,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f"\n  {nom}")
    print(f"     CV R²  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}  "
          f"(folds: {[f'{v:.3f}' for v in cv_r2]})")
    print(f"     CV MAE : {(-cv_mae).mean():.2f} ± {(-cv_mae).std():.2f} kg/ha")

# ── 7. Champion ───────────────────────────────────────────────────────────────
resultats      = compare_models(dictionnaire_modeles, X_test_sel, y_test)
nom_gagnant    = resultats["winner"]
modele_gagnant = resultats["winner_model"]
preds_test     = modele_gagnant.predict(X_test_sel)
r2             = r2_score(y_test, preds_test)
mae            = mean_absolute_error(y_test, preds_test)
rmse           = np.sqrt(mean_squared_error(y_test, preds_test))
wmape          = resultats["scores"][nom_gagnant]["wMAPE"]
ecart_final    = r2_score(y_train, modele_gagnant.predict(X_train_sel)) - r2

print(f"\n{SEP}")
print("  RAPPORT FINAL — MODÈLE TEMPOREL v15.4")
print(SEP)
print(f"  Modèle Champion : {nom_gagnant}")
print(f"  R²  Test        : {r2:.4f}   {'✓ BON' if r2 >= 0.45 else '△ ACCEPTABLE' if r2 >= 0.28 else '⚠ FAIBLE'}")
print(f"  Écart train/test: {ecart_final:.4f}  {'✓ OK' if ecart_final < 0.25 else '⚠ Overfitting résiduel'}")
print(f"  MAE Test        : {mae:.2f} kg/ha")
print(f"  RMSE Test       : {rmse:.2f} kg/ha")
print(f"  wMAPE           : {wmape:.2f}%")
print(SEP)

if hasattr(modele_gagnant, 'feature_importances_'):
    imp2 = modele_gagnant.feature_importances_
    print("\n  Importance des features (modèle champion) :")
    print(f"  {'Feature':<38} Importance  Cum.")
    print(f"  {'-'*55}")
    cumul = 0
    for i, idx in enumerate(np.argsort(imp2)[::-1]):
        cumul += imp2[idx]
        print(f"  {i+1:2d}. {features_sel[idx]:<35} {imp2[idx]:.4f}    {cumul:.4f}")

# ── 8. Exportation ────────────────────────────────────────────────────────────
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT : extension .joblib (cohérent avec prediction_service.py)
joblib.dump(modele_gagnant, config.OUTPUT_DIR / "meilleur_modele_temporel.joblib")
joblib.dump(scaler_X,       config.OUTPUT_DIR / "scaler_temporel.joblib")

with open(config.OUTPUT_DIR / "features_temporel.txt", "w", encoding="utf-8") as f:
    f.write(",".join(features_sel))

global_stats = {
    'mean': float(df_train_raw['Rendement_kg_ha'].mean()),
    'std':  float(df_train_raw['Rendement_kg_ha'].std()),
    'mean_temp_max_historique': mean_temp_max_train
}
with open(config.OUTPUT_DIR / "statistiques_globales.json", "w", encoding="utf-8") as f:
    json.dump(global_stats, f, indent=2, ensure_ascii=False)

print(f"\n✓ Artefacts temporels v15.4 exportés → {config.OUTPUT_DIR}")

# ── 9. Graphiques ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=config.DPI)

axes[0].scatter(y_test, preds_test, color="mediumvioletred", alpha=0.6, edgecolors='k', s=50)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel("Rendements Observés (kg/ha)")
axes[0].set_ylabel("Rendements Estimés (kg/ha)")
axes[0].set_title(f"Temporel v15.4 — {nom_gagnant}\nR²={r2:.3f} | MAE={mae:.0f} kg/ha")
axes[0].grid(True, linestyle="--", alpha=0.4)

residus = preds_test - y_test
axes[1].hist(residus, bins=25, color="darkorange", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', lw=2)
axes[1].axvline(residus.mean(), color='gold', linestyle='--', lw=1.5,
                label=f"Moy={residus.mean():.0f}")
axes[1].set_xlabel("Résidu (kg/ha)")
axes[1].set_title(f"Résidus — Moy={residus.mean():.1f} | Std={residus.std():.1f}")
axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.4)

if hasattr(modele_gagnant, 'feature_importances_'):
    top10 = np.argsort(modele_gagnant.feature_importances_)[::-1][:10]
    axes[2].barh([features_sel[i] for i in top10][::-1],
                  modele_gagnant.feature_importances_[top10][::-1],
                  color="mediumvioletred", edgecolor="white")
    axes[2].set_xlabel("Importance")
    axes[2].set_title("Top 10 Features Pédoclimatiques")
    axes[2].grid(True, axis='x', linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(config.OUTPUT_DIR / "evaluation_temporelle_v15.png", bbox_inches='tight')
plt.close()
print("✓ Graphiques sauvegardés.")