import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import config
from data_loader import load_and_clean
from feature_engineering import create_all_features
warnings.filterwarnings('ignore')
print("\n" + "=" * 80)
print("  ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
print("=" * 80)

# CHARGEMENT ET FEATURES
print("\n Chargement des données...")
df_raw = load_and_clean()

print("\n Engineering des features...")
df = create_all_features(df_raw)

#  ANALYSES VISUELLES
print("\n Génération des visualisations...")

# Statistiques descriptives
cols_num = df.select_dtypes(include=[np.number]).columns
stats = df[cols_num].describe().T
stats['skewness'] = df[cols_num].skew()
stats['kurtosis'] = df[cols_num].kurt()
stats.to_csv(config.EDA_DIR / "rapport_statistiques_univariees.csv")
print(f" Statistiques univariées exportées")

# Plot 01 : Distribution du rendement
plt.figure(figsize=(10, 5))
sns.histplot(df[config.TARGET_COL], kde=True, color="darkgreen", bins=20, edgecolor='black', alpha=0.7)
plt.axvline(df[config.TARGET_COL].mean(), color='red', linestyle='--', linewidth=2, 
            label=f"Moyenne : {df[config.TARGET_COL].mean():.2f}")
plt.axvline(df[config.TARGET_COL].median(), color='blue', linestyle=':', linewidth=2, 
            label=f"Médiane : {df[config.TARGET_COL].median():.2f}")
plt.title("Distribution des Rendements Cotonniers", fontsize=12, fontweight='bold')
plt.xlabel("Rendement_kg_ha")
plt.ylabel("Densité")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(config.EDA_DIR / "01_distribution_rendement.png", dpi=config.DPI)
plt.close()
print(" Plot 01 : Distribution")

# Plot 02 : Variabilité géographique
plt.figure(figsize=(12, 6))
ordre = df.groupby('Commune')[config.TARGET_COL].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='Commune', y=config.TARGET_COL, order=ordre, palette="viridis")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title("Hétérogénéité des Rendements par Commune", fontsize=12, fontweight='bold')
plt.xlabel("Communes")
plt.ylabel("Rendement_kg_ha")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(config.EDA_DIR / "02_variabilite_geographique.png", dpi=config.DPI)
plt.close()
print(" Plot 02 : Variabilité géographique")

# Plot 03 : Trajectoire temporelle
plt.figure(figsize=(10, 5))
df_time = df.copy()
df_time['Annee'] = df_time['Annee'].astype(int)
sns.lineplot(data=df_time, x='Annee', y=config.TARGET_COL, marker='o', color='darkorange', 
             linewidth=2.5, errorbar=('ci', 95))
plt.title("Trajectoire Historique des Rendements", fontsize=12, fontweight='bold')
plt.xlabel("Campagne Agricole")
plt.ylabel("Rendement Moyen (kg/ha)")
plt.xticks(df_time['Annee'].unique(), rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(config.EDA_DIR / "03_evolution_temporelle.png", dpi=config.DPI)
plt.close()
print(" Plot 03 : Évolution temporelle")

# Plot 04 : Matrice de corrélation
plt.figure(figsize=(15, 13))
corr_matrix = df[cols_num].corr(method='pearson')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": .7}, annot_kws={"size": 7})
plt.title("Matrice de Corrélations Multi-Variées", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(config.EDA_DIR / "04_matrice_correlation_interne.png", dpi=config.DPI)
plt.close()
print(" Plot 04 : Matrice de corrélation")

# Plot 05 : Sensibilité au rendement
corr_target = corr_matrix[config.TARGET_COL].drop(config.TARGET_COL).sort_values(ascending=False)
corr_target.to_csv(config.EDA_DIR / "liaisons_directes_rendement.csv")
plt.figure(figsize=(11, 7))
colors = ['teal' if x >= 0 else 'crimson' for x in corr_target.values]
sns.barplot(x=corr_target.values, y=corr_target.index, palette=colors, hue=corr_target.index, legend=False)
plt.axvline(0, color='black', linewidth=1.2)
plt.title("Sensibilité Linéaire au Rendement", fontsize=12, fontweight='bold')
plt.xlabel("Corrélation de Pearson")
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(config.EDA_DIR / "05_sensibilite_directe_facteurs.png", dpi=config.DPI)
plt.close()
print(" Plot 05 : Sensibilité")

print(f"\n EDA complétée - Fichiers dans : {config.EDA_DIR}")
print("=" * 80)
