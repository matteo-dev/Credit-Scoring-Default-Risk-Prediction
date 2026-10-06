# Fichier python pour l'entraînement et l'évaluation du modèle XGBoost avec Optuna pour l'optimisation des hyperparamètres.

# Importation des bibliothèques nécessaires
import pandas as pd
import xgboost as xgb # Bibliothèque pour le modèle XGBoost
import optuna # Bibliothèque pour l'optimisation des hyperparamètres
import joblib # Bibliothèque pour la sérialisation des objets Python (sauvegarde du modèle)
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc # Bibliothèque pour les métriques d'évaluation
from scipy.stats import ks_2samp # Bibliothèque pour le test de Kolmogorov-Smirnov (KS)

# Fonction pour calculer la statistique KS (Kolmogorov-Smirnov) entre les distributions des probabilités prédites pour les classes positives et négatives.
def calculate_ks_statistic(y_true, y_proba):

    # Séparer les probabilités des cibles positives (défaut) et négatives (sain)
    proba_default = y_proba[y_true == 1]
    proba_non_default = y_proba[y_true == 0]

    # Test KS à deux échantillons
    ks_stat, p_value = ks_2samp(proba_default, proba_non_default)
    return ks_stat

# Fonction objectif pour Optuna : elle définit l'espace de recherche des hyperparamètres et retourne la métrique à maximiser (ROC-AUC).
def objective(trial, X_train, y_train):

    # Espace de recherche des hyperparamètres XGBoost
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # Gestion du déséquilibre de classes (poids sur la classe minoritaire)
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
    }
    
    # Entraînement simplifié pour l'optimisation (utilisation d'un split de validation)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
    
    # Entraînement du modèle XGBoost avec les hyperparamètres proposés par Optuna
    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    # Évaluation sur le jeu de validation
    preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, preds)
    return auc_score

# Fonction principale pour entraîner le modèle XGBoost avec Optuna et évaluer ses performances sur le jeu de test.
def train_and_evaluate():

    print("Chargement des données d'entraînement...")
    X_train = pd.read_parquet('X_train.parquet')
    X_test = pd.read_parquet('X_test.parquet')
    y_train = pd.read_parquet('y_train.parquet')['TARGET']
    y_test = pd.read_parquet('y_test.parquet')['TARGET']
    
    print("Lancement de l'optimisation Optuna...")
    study = optuna.create_study(direction='maximize')

    # Limité à 10 essais pour l'exemple (à augmenter en production)
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10)
    
    print("Meilleurs paramètres trouvés : ", study.best_params)
    
    # Entraînement du modèle final avec les meilleurs hyperparamètres trouvés
    print("Entraînement du modèle final...")
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    
    # Entraînement du modèle final sur l'ensemble du jeu d'entraînement
    final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
    final_model.fit(X_train, y_train)
    
    # Évaluation du modèle final sur le jeu de test
    print("Évaluation sur le jeu de test...")
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    
    # 1. ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 2. PR-AUC (Area Under Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # 3. KS Stat
    ks_stat = calculate_ks_statistic(y_test, y_pred_proba)
    
    print(f"--- Performances Bancaires ---")
    print(f"ROC-AUC : {roc_auc:.4f} (Discrimination globale)")
    print(f"PR-AUC  : {pr_auc:.4f} (Focus sur les défauts)")
    print(f"KS Stat : {ks_stat:.4f} (Séparation des distributions)")
    
    # Sauvegarde du modèle à la racine
    joblib.dump(final_model, 'xgboost_model.joblib')
    print("Modèle sauvegardé sous 'xgboost_model.joblib'")

if __name__ == "__main__":
    train_and_evaluate()