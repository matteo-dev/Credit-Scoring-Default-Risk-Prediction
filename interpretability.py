# Fichier python pour l'interprétabilité du modèle XGBoost avec SHAP

# Importation des bibliothèques nécessaires
import pandas as pd
import xgboost as xgb
import shap # Bibliothèque pour l'explicabilité des modèles (SHAP)
import joblib
import matplotlib.pyplot as plt

# Fonction pour générer un explainer SHAP global pour le modèle XGBoost
def generate_global_shap_explainer():

    print("Chargement du modèle et des données pour SHAP...")

    # Chargement du modèle XGBoost entraîné et des données d'entraînement
    model = joblib.load('xgboost_model.joblib')
    X_train = pd.read_parquet('X_train.parquet')
    
    print("Création de l'explainer SHAP (TreeExplainer)...")

    # TreeExplainer est optimisé pour les modèles à base d'arbres (XGBoost, LightGBM)
    explainer = shap.TreeExplainer(model)
    
    # Sauvegarde de l'explainer à la racine
    joblib.dump(explainer, 'shap_explainer.joblib')
    print("Explainer SHAP sauvegardé sous 'shap_explainer.joblib'")
    
    return explainer

# Fonction pour générer un graphique global de l'importance des variables avec SHAP
def plot_shap_summary(X_sample):

    # Chargement de l'explainer SHAP sauvegardé
    explainer = joblib.load('shap_explainer.joblib')
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    print("Summary Plot sauvegardé sous 'shap_summary_plot.png'")

# Fonction pour générer un Waterfall Plot pour un client spécifique
def explain_local_prediction(client_data_processed):

    # Chargement de l'explainer SHAP sauvegardé
    explainer = joblib.load('shap_explainer.joblib')
    
    # Calcul de la valeur SHAP pour l'individu
    shap_values = explainer(client_data_processed)
    
    plt.figure()

    # Affichage en cascade (waterfall) de l'apport de chaque variable par rapport à l'espérance du modèle
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()

    # Retourne la figure pour pouvoir l'afficher dans Streamlit par exemple
    return plt.gcf()

if __name__ == "__main__":
    
    # Initialisation de l'explainer
    explainer = generate_global_shap_explainer()
    
    # Test de génération du graphique global sur un échantillon (pour éviter un temps de calcul trop long)
    X_test = pd.read_parquet('X_test.parquet')
    X_sample = X_test.sample(100, random_state=42)
    plot_shap_summary(X_sample)