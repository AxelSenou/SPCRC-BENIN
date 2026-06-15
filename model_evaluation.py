
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X: np.ndarray, y_true: np.ndarray) -> dict:
   
    try:
        preds = model.predict(X)
        
        r2 = r2_score(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        wmape = (np.sum(np.abs(y_true - preds)) / np.sum(np.abs(y_true))) * 100
        biais = np.mean(preds - y_true)
        
        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "wMAPE": wmape,
            "Biais": biais,
            "preds": preds
        }
    except Exception as e:
        raise ValueError(f"Erreur lors de l'évaluation : {str(e)}")

def compare_models(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
   
    scores = {}
    for name, model in models_dict.items():
        scores[name] = evaluate_model(model, X_test, y_test)
    
    # Sélectionner le meilleur par R²
    best_name = max(scores.keys(), key=lambda k: scores[k]["R2"])
    
    return {
        "winner": best_name,
        "scores": scores,
        "winner_model": models_dict[best_name]
    }

def print_benchmark_report(scores: dict) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK DES MODÈLES")
    print("=" * 80)
    
    for name, metrics in scores.items():
        print(f"\n{name:15} → R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} kg/ha | "
              f"MAE: {metrics['MAE']:.4f} kg/ha | wMAPE: {metrics['wMAPE']:.2f}% | "
              f"Biais: {metrics['Biais']:+.4f}")
    
    print("=" * 80)
