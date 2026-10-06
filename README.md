# Credit Scoring & Explainability 


## Description (Français)
Ce projet déploie une architecture complète de Machine Learning pour l'évaluation du risque de crédit bancaire. Il modélise la probabilité de défaut (PD) d'un emprunteur et intègre le calcul de la perte attendue (Expected Loss : EL = PD x LGD x EAD). 

Face aux exigences réglementaires strictes du secteur bancaire (Model Risk Management, accords de Bâle, RGPD), le projet met un point d'honneur sur la transparence. Chaque décision automatisée est justifiable et auditable grâce à l'explicabilité SHAP.

Ce portfolio a été développé en appliquant les standards de production métier, en parallèle d'un cursus de Master 2 en mathématiques appliquées et finance quantitative à CY Tech et d'une pratique quotidienne en Data Analytics.

## Architecture du Projet
* `preprocess.py` : Nettoyage des données, imputation et ingénierie des caractéristiques financières (ratios d'endettement, stabilité).
* `train.py` : Entraînement d'un modèle XGBoost optimisé via Optuna. Évaluation avec des métriques bancaires (ROC-AUC, PR-AUC, Statistique KS).
* `interpretability.py` : Génération des valeurs SHAP pour l'analyse globale et locale.
* `app.py` : Interface web Streamlit interactive permettant la saisie d'un dossier client, le calcul de la probabilité de défaut, l'affichage de l'Expected Loss et la génération du Waterfall Plot SHAP.

## Utilisation
1. Installer les dépendances : `pip install -r requirements.txt`
2. Préparer les données (nécessite le dataset Kaggle Home Credit à la racine) : `python preprocess.py`
3. Entraîner le modèle : `python train.py`
4. Lancer le tableau de bord décisionnel : `streamlit run app.py`

---

## Description (English)
This project deploys a complete Machine Learning architecture for bank credit risk assessment. It models a borrower's Probability of Default (PD) and incorporates the Expected Loss calculation (EL = PD x LGD x EAD).

Given the strict regulatory requirements in the banking sector (Model Risk Management, Basel Accords, GDPR), this project prioritizes transparency. Every automated decision is justifiable and auditable using SHAP explainability. 

This portfolio piece was developed applying industry production standards, running alongside a Master's degree in applied mathematics and quantitative finance at CY Tech and daily practice in Data Analytics.

## Project Architecture
* `preprocess.py`: Data cleaning, imputation, and financial feature engineering (debt-to-income ratios, stability metrics).
* `train.py`: Training an XGBoost model optimized with Optuna. Evaluated using core banking metrics (ROC-AUC, PR-AUC, KS Statistic).
* `interpretability.py`: Generation of SHAP values for global and local interpretability.
* `app.py`: Interactive Streamlit web interface allowing client data input, computing default probability and Expected Loss, and generating the SHAP Waterfall Plot.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare the data (requires the Kaggle Home Credit dataset in the root directory): `python preprocess.py`
3. Train the model: `python train.py`
4. Launch the decision dashboard: `streamlit run app.py`
