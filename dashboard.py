# Fichier python pour le dashboard Streamlit de scoring de crédit avec prédiction du risque de défaut et interprétabilité du modèle.

# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# 1. CHARGEMENT DES RESSOURCES (Mise en cache pour les performances)

# Utilisation de la mise en cache pour éviter de recharger les modèles et le préprocesseur à chaque interaction
@st.cache_resource

# Fonction pour charger le préprocesseur, le modèle et l'explainer SHAP depuis les fichiers sauvegardés
def load_assets():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('xgboost_model.joblib')
    
    # Remplacer joblib.load par la création directe de l'explainer SHAP
    explainer = shap.TreeExplainer(model)
    
    return preprocessor, model, explainer

# 2. CONFIGURATION DE LA PAGE STREAMLIT

st.set_page_config(page_title="Credit Scoring Dashboard", page_icon="🏦", layout="wide")
st.title("🏦 Credit Scoring & Default Risk Prediction")
st.markdown("Enter the applicant's details in the sidebar to predict their default risk and understand the model's decision.")


# 3. PANNEAU LATÉRAL : SAISIE DES DONNÉES CLIENT (Sidebar)

st.sidebar.header("📋 Applicant Information")

# Saisie des informations socio-démographiques
gender = st.sidebar.selectbox("Gender", ["M", "F", "XNA"])
age = st.sidebar.slider("Age (Years)", 18, 90, 35)
employed_years = st.sidebar.slider("Years of Employment", 0.0, 50.0, 5.0)
education = st.sidebar.selectbox("Education Level", [
    "Secondary / secondary special", 
    "Higher education", 
    "Incomplete higher", 
    "Lower secondary", 
    "Academic degree"
])

# Saisie des informations financières
st.sidebar.markdown("---")
st.sidebar.header("💰 Financial Information")
contract_type = st.sidebar.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
income = st.sidebar.number_input("Total Annual Income ($)", min_value=10000, max_value=10000000, value=50000, step=5000)
credit_amount = st.sidebar.number_input("Requested Credit Amount ($)", min_value=10000, max_value=5000000, value=150000, step=10000)
annuity = st.sidebar.number_input("Yearly Annuity ($)", min_value=1000, max_value=500000, value=10000, step=1000)

# Scores externes (issus de bases de données de crédit externes)
st.sidebar.markdown("---")
st.sidebar.header("📊 External Credit Scores")
ext_source_1 = st.sidebar.slider("External Source 1", 0.0, 1.0, 0.5)
ext_source_2 = st.sidebar.slider("External Source 2", 0.0, 1.0, 0.5)
ext_source_3 = st.sidebar.slider("External Source 3", 0.0, 1.0, 0.5)


# 4. PRÉPARATION DE LA DONNÉE POUR LE MODÈLE

# Reconstruction des features brutes comme attendu par le preprocessor
raw_data = pd.DataFrame({
    'CODE_GENDER': [gender],
    'DAYS_BIRTH': [-age * 365],
    'DAYS_EMPLOYED': [-employed_years * 365],
    'NAME_EDUCATION_TYPE': [education],
    'NAME_CONTRACT_TYPE': [contract_type],
    'AMT_INCOME_TOTAL': [income],
    'AMT_CREDIT': [credit_amount],
    'AMT_ANNUITY': [annuity],
    'EXT_SOURCE_1': [ext_source_1],
    'EXT_SOURCE_2': [ext_source_2],
    'EXT_SOURCE_3': [ext_source_3]
})

# Recalcul des ratios métiers à la volée
raw_data['CREDIT_INCOME_RATIO'] = raw_data['AMT_CREDIT'] / raw_data['AMT_INCOME_TOTAL']
raw_data['ANNUITY_INCOME_RATIO'] = raw_data['AMT_ANNUITY'] / raw_data['AMT_INCOME_TOTAL']
raw_data['AGE_YEARS'] = age
raw_data['EMPLOYED_YEARS'] = employed_years

# Variables strictement nécessaires pour le pipeline
features = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 
    'AGE_YEARS', 'EMPLOYED_YEARS',
    'AMT_CREDIT', 'AMT_INCOME_TOTAL',
    'NAME_EDUCATION_TYPE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE'
]
X_client = raw_data[features]

# Application du preprocessing
X_client_processed = preprocessor.transform(X_client)

# Récupération des noms de colonnes pour SHAP
cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['NAME_EDUCATION_TYPE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE'])
numeric_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'AGE_YEARS', 'EMPLOYED_YEARS', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']
feature_names = numeric_features + list(cat_feature_names)
X_client_final = pd.DataFrame(X_client_processed, columns=feature_names)


# 5. PRÉDICTION ET AFFICHAGE (Jauge, Expected Loss & Explicabilité)

# Calcul de la probabilité de défaut (PD)
default_probability = model.predict_proba(X_client_final)[0, 1]
threshold = 0.40 
is_approved = default_probability < threshold

# LGD (Loss Given Default) : Fixée à 45% (standard réglementaire)
LGD = 0.45 

# EAD (Exposure At Default) : Montant du crédit + majoration de 10% (intérêts/pénalités)
EAD = credit_amount * 1.10

# Calcul de l'Expected Loss (EL = PD * LGD * EAD)
expected_loss = default_probability * LGD * EAD

st.markdown("---")
col1, col2 = st.columns([1, 2])

# Graphique de jauge pour la probabilité de défaut et affichage de la décision de crédit
with col1:
    st.subheader("Credit Decision")
    
    if is_approved:
        st.success("✅ **APPROVED** - Low risk of default.")
    else:
        st.error("❌ **REJECTED** - High risk of default.")
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = default_probability * 100,
        title = {'text': "Default Probability (PD) %"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': "lightgreen"},
                {'range': [threshold * 100, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Affichage de l'Expected Loss
    st.markdown(f"### 📉 Expected Loss (EL): **${expected_loss:,.2f}**")
    st.caption(f"Calculated as: PD ({default_probability:.2%}) × LGD ({LGD:.0%}) × EAD (${EAD:,.0f})")

# Graphique d'interprétabilité SHAP pour expliquer la décision du modèle
with col2:
    st.subheader("Model Interpretability (SHAP)")
    st.markdown("This waterfall plot explains the specific reasons for this decision. It shows how each piece of client data pushes the default probability higher (red) or lower (blue) compared to the average applicant.")
    
    shap_values = explainer(X_client_final)
    
    fig_shap, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig_shap)
