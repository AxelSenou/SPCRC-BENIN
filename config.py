from pathlib import Path
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    
    
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT   
    OUTPUT_DIR: Path = PROJECT_ROOT / "rapport"
    EDA_DIR: Path = PROJECT_ROOT / "eda"
    
    RAW_DATASET: str = "dataset_meteo_stress_benin.csv"
    CLEANED_DATASET: str = "dataset_nettoye.csv"
    SOIL_DATASET: str = "dataset_sol_benin.csv"
    
    MODEL_PATH: str = "meilleur_modele_spcrc.json"
    SCALER_PATH: str = "scaler_spcrc.json"
    ENCODING_MAPPING_PATH: str = "mapping_communes_target_encoding.json"
    FEATURES_PATH: str = "structure_features.txt"
    
    TARGET_COL: str = "Rendement_kg_ha"
    CATEGORICAL_COLS: List[str] = field(default_factory=lambda: ['Commune', 'Annee'])
    
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    DPI: int = 150
    FIGURE_SIZE: tuple = field(default_factory=lambda: (10, 5))
    
    M_SMOOTHING: int = 5
    STRESS_THRESHOLD: float = 1.0
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.EDA_DIR.mkdir(parents=True, exist_ok=True)

config = Config()