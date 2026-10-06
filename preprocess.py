# Fichier python pour le preprocessing des données du dataset Kaggle "Home Credit Default Risk"

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # Bibliothèque pour l'imputation des valeurs manquantes
from sklearn.preprocessing import StandardScaler, OneHotEncoder #Bibliothèque pour la normalisation et l'encodage des variables catégorielles
from sklearn.compose import ColumnTransformer # Bibliothèque pour appliquer différentes transformations à différentes colonnes
from sklearn.pipeline import Pipeline # Bibliothèque pour créer des pipelines de transformation 
import joblib # Bibliothèque pour la sérialisation des objets Python (sauvegarde du pipeline)

# Fonction principale pour charger et prétraiter les données
def load_and_preprocess_data(filepath="application_train.csv"):

    print("Chargement des données...")

    # Limiter le nombre de lignes pour les tests initiaux si nécessaire
    df = pd.read_csv(filepath)
    
    # 1. Feature Engineering Métier

    # Création du Ratio d'endettement (Debt-to-Income / Credit-to-Income)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # Ratio d'annuité (Charge mensuelle sur revenus)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # Stabilité (Âge en années vs Ancienneté professionnelle)
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True) # Valeur aberrante classique du dataset
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365
    df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'] / -365
    
    # Séparation Features (X) et Target (y)
    y = df['TARGET']

    # Sélection d'un sous-ensemble pertinent pour l'exemple
    features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
        'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 
        'AGE_YEARS', 'EMPLOYED_YEARS',
        'AMT_CREDIT', 'AMT_INCOME_TOTAL',
        'NAME_EDUCATION_TYPE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE'
    ]

    X = df[features]
    
    # 2. Séparation Train / Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Pipelines de transformation
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Pipeline pour variables numériques : Imputation par la médiane + Standardisation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline pour variables catégorielles : Imputation par le mode + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinaison des pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Application du preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Récupération des noms de colonnes après One-Hot Encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = list(numeric_features) + list(cat_feature_names)
    
    # Transformation en DataFrame pour compatibilité avec SHAP plus tard
    X_train_final = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # 4. Sauvegarde du pipeline pour l'application Streamlit
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("Pipeline de preprocessing sauvegardé sous 'preprocessor.joblib'")
    
    return X_train_final, X_test_final, y_train, y_test

if __name__ == "__main__":

    # Exécution standalone pour préparer les données
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Sauvegarde des sets d'entraînement/test pour l'étape de modélisation
    X_train.to_parquet('X_train.parquet')
    X_test.to_parquet('X_test.parquet')
    y_train.to_frame().to_parquet('y_train.parquet')
    y_test.to_frame().to_parquet('y_test.parquet')
    print("Fichiers de données préparées sauvegardés à la racine.")