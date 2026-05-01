#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:56:35 2026

@author: hounsousamuel
"""

import os
import sys
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tempfile
import traceback
from datetime import datetime
from main import main

class Interface:
    def __init__(self):
        self.th = None

    def __call__(self):
        if "history" not in st.session_state:
            st.session_state.result = None
            st.session_state.model = None
            st.session_state.running = False
        
        st.set_page_config(
            page_title="SmartPredictor - AutoML Intelligent",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .warning-box {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left: 5px solid #ff9800;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .danger-box {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 5px solid #f44336;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .success-box {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 5px solid #4caf50;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .info-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 5px solid #2196f3;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🤖 SmartPredictor</h1>
            <p style="color: white; margin: 5px 0 0 0;">Auto-ML · Détecte ton problème et entraîne le bon modèle</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("📅 by @hounsoubenny-cyber")
        
        # ⚠️ WARNING IMPORTANT sur les problèmes de régression
        st.markdown("""
        <div class="danger-box">
            <strong>⚠️ ATTENTION - LIMITES DU MODÈLE DE RÉGRESSION</strong><br><br>
            📊 <strong>Ce modèle peut mal deviner !</strong> Les prédictions de régression sont basées sur des corrélations statistiques 
            et peuvent être inexactes, surtout si :<br>
            • Les données d'entraînement ne sont pas représentatives<br>
            • Vous prédisez en dehors de la plage des données d'entraînement (extrapolation)<br>
            • Des variables importantes manquent dans le jeu de données<br>
            • Les relations ne sont pas linéaires ou suivent des motifs complexes<br><br>
            🎯 <strong>Recommandations :</strong><br>
            • Visualisez toujours les résidus et les métriques d'erreur (MAE, RMSE, R²)<br>
            • Comparez les prédictions avec des valeurs réelles si disponibles<br>
            • Utilisez la validation croisée pour évaluer la stabilité du modèle<br>
            • Méfiez-vous des prédictions pour de nouvelles observations trop différentes<br>
            • Pour des décisions critiques, faites valider par un expert domaine<br><br>
            <em>⚠️ Les résultats de régression sont des estimations avec une marge d'erreur potentielle.</em>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            st.divider()
            
            optimize = st.toggle("🔬 Bayesian Optimization", value=True, key="opt_toggle", help="Optimise les hyperparamètres automatiquement")
            
            with st.container(border=True):
                max_iter = 1
                if optimize:
                    max_iter = st.number_input(
                        "🔄 Itérations max d'optimisation :", 
                        max_value=200, 
                        min_value=1, 
                        key="max_iter", 
                        value=10,
                        help="Plus d'itérations = meilleur modèle mais plus lent"
                    )
                
                what = st.selectbox(
                    "🎯 Action à effectuer :", 
                    options=["fit_predict", "predict", "fit"], 
                    key="action_select",
                    help="fit: entraînement seul | predict: prédiction seule | fit_predict: les deux"
                )
                
                problem_type = st.selectbox(
                    "📊 Type de problème :",
                    options=["binary", 'continuous', 'multiclass', 'multilabel', "explain", "deviner"],
                    key="problem_type",
                    help="continuous = régression | binary/multiclass = classification"
                )
                
                smote = False
                if what == "fit":
                    smote = st.selectbox("⚖️ Appliquer SMOTE (équilibrage)", options=["True", "False"], key="smote_select") == "True"

                model_path = st.text_input(
                    "💾 Chemin de sauvegarde du modèle", 
                    value="model.pkl", 
                    key="model_path",
                    help="Où sauvegarder le modèle entraîné"
                ).strip()

                verbose = st.selectbox("🔊 Verbosité", options=["True", "False"], key="verbose") == "True"
            
            st.divider()
            st.markdown("""
            <div class="info-box" style="font-size: 0.8em;">
                <strong>💡 Astuces :</strong><br>
                • Pour la régression, vérifiez le R²<br>
                • Plus de données = meilleures prédictions<br>
                • Les features pertinentes sont cruciales
            </div>
            """, unsafe_allow_html=True)

        file = st.file_uploader(
            "📂 Uploader vos données :", 
            type=["csv", "json", "pkl", "joblib", "xls", "xlsx"], 
            key="file_uploader", 
            accept_multiple_files=False,
            help="Formats supportés: CSV, JSON, PKL, JOBLIB, Excel"
        )
        
        if file:
            try:
                ext = os.path.splitext(file.name)[-1].lower()
                
                readers = {
                    ".pkl": pd.read_pickle,
                    ".joblib": joblib.load,
                    ".json": pd.read_json,
                    ".xls": pd.read_excel,
                    ".xlsx": pd.read_excel,
                    ".csv": pd.read_csv
                }
                
                func = readers.get(ext, pd.read_csv)

                dataset_path = ""
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    dataset_path = temp_file.name

                final_path = list(os.path.split(dataset_path))
                final_path[-1] = file.name
                final_path = os.path.join(*final_path)
                os.rename(dataset_path, final_path)
                dataset_path = final_path
                
                data = pd.DataFrame(func(dataset_path))
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    apercu_size = st.number_input(
                        "📋 Taille de l'aperçu", 
                        min_value=1, 
                        max_value=len(data) - 1, 
                        key="preview_size", 
                        value=min(10, len(data))
                    )
                    st.write(f"**Aperçu des {apercu_size} premières lignes :**")
                    st.dataframe(data.head(apercu_size), width='stretch')
                
                with col2:
                    st.metric("📊 Dimensions", f"{data.shape[0]:,} × {data.shape[1]}")
                    st.metric("🔢 Colonnes", len(data.columns))
                    st.metric("💾 Mémoire", f"{data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

                target_name = st.selectbox(
                    "🎯 Nom de la target (variable à prédire) :", 
                    options=list(data.columns), 
                    key="target_select",
                    help="Colonne contenant les valeurs à prédire"
                )
                
                CONFIG = {
                    "target_name": target_name,
                    "problem_type": problem_type,
                    "model_name": model_path,
                    "what": what,
                    "v": verbose,
                    "opt": optimize,
                    "max_iter": max_iter,
                    "dataset_path": dataset_path,
                    "smote": smote,
                }

                # Bouton de démarrage
                col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
                with col_btn1:
                    start_button = st.button(
                        "🚀 Démarrer l'analyse !", 
                        type="primary", 
                        icon=":material/play_arrow:", 
                        width='stretch', 
                        key="start_btn"
                    )
                
                if start_button:
                    st.session_state.running = True
                    with st.spinner(text="🔍 Analyse en cours, veuillez patienter...", show_time=True):
                        try:
                            st.info("📝 Démarrage de l'analyse - Ne plus modifier l'interface")
                            
                            with st.expander("⚙️ Configuration utilisée", expanded=False):
                                config_df = pd.DataFrame({k: [str(v)] for k, v in CONFIG.items()}).T
                                config_df.columns = ["Valeur"]
                                st.dataframe(config_df, width='stretch')
                            
                            if CONFIG["what"] != "fit_predict":
                                returned = main(CONFIG, include_model=True)
                                if returned and "model" in returned:
                                    model = returned["model"]
                                else:
                                    model = None
                            else:
                                CONF_COPY = CONFIG.copy()
                                CONF_COPY["what"] = "fit"
                                st.info("🏋️ Phase 1: Entraînement du modèle...")
                                returned = main(CONF_COPY, include_model=True)
                                CONF_COPY["what"] = "predict"
                                st.info("🔮 Phase 2: Prédictions sur les données...")
                                model = returned["model"] if returned and "model" in returned else None
                                returned = main(CONF_COPY, include_model=True)

                            st.session_state.model = model
                            st.session_state.result = returned
                            st.session_state.running = False
                            
                        except Exception as e:
                            st.error(f"❌ Erreur pendant l'exécution : {str(e)}")
                            st.code(traceback.format_exc(), language="python")
                            st.session_state.running = False

                # Affichage des résultats
                if st.session_state.result and not st.session_state.running:
                    returned = st.session_state.result
                    model = st.session_state.model
                    
                    if returned and returned.get("errors", []):
                        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                        st.write("❌ **Erreurs survenues :**")
                        for err in returned["errors"]:
                            st.error(err)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Succès
                    if returned and not returned.get("errors"):
                        if what == "fit":
                            st.markdown("""
                            <div class="success-box">
                                ✅ <strong>Entraînement réussi !</strong><br>
                                Modèle sauvegardé dans: <code>{}</code>
                            </div>
                            """.format(returned.get('model', {}).model_path if returned.get('model') else model_path), unsafe_allow_html=True)
                        elif what == "predict":
                            st.markdown('<div class="success-box">✅ <strong>Prédictions réalisées avec succès !</strong></div>', unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box">
                                ✅ <strong>Processus complet réussi !</strong><br>
                                • Modèle entraîné et sauvegardé<br>
                                • Prédictions générées
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Historique des modèles
                    if model and hasattr(model, 'history') and model.history:
                        st.divider()
                        st.markdown("### 🧠 Modèles entraînés")
                        models_trained = model.history.get('models', [])
                        st.info(f"📊 {len(models_trained)} modèle(s) entraîné(s) : {', '.join(models_trained)}")
                    
                    if model and hasattr(model, 'history') and model.history.get("opt", {}):
                        st.markdown("### 🔬 Résultats des optimisations bayésiennes")
                        tabs = st.tabs(list(model.history["opt"].keys()))
                        for tab, (model_name, opt_results) in zip(tabs, model.history["opt"].items()):
                            with tab:
                                st.markdown(f"**Modèle : {model_name}**")
                                cols = st.columns(min(len(opt_results), 3))
                                for idx, (param_name, param_value) in enumerate(opt_results.items()):
                                    with cols[idx % 3]:
                                        if isinstance(param_value, pd.DataFrame):
                                            st.dataframe(param_value, width='stretch')
                                        else:
                                            st.metric(param_name, value=param_value)
                    
                    # Résultats d'évaluation
                    if model and hasattr(model, 'history') and model.history.get("eval", {}):
                        st.markdown("### 📈 Résultats d'évaluation")
                        eval_data = model.history["eval"]
                        
                        if problem_type == "continuous":
                            st.markdown("""
                            <div class="warning-box">
                                <strong>📉 Interprétation des métriques de régression :</strong><br>
                                • <strong>R² (coefficient de détermination)</strong> : Entre 0 et 1 - plus proche de 1, meilleur le modèle<br>
                                • <strong>MAE (Erreur Absolue Moyenne)</strong> : Erreur moyenne en unités originales - plus petite = meilleure<br>
                                • <strong>RMSE (Racine de l'Erreur Quadratique Moyenne)</strong> : Pénalise les grandes erreurs<br>
                                • <strong>MAPE (%)</strong> : Erreur relative - idéal < 10% pour un bon modèle
                            </div>
                            """, unsafe_allow_html=True)
                        
                        for metric_name, metric_value in eval_data.items():
                            if isinstance(metric_value, pd.DataFrame):
                                st.subheader(metric_name.replace('_', ' ').title())
                                st.dataframe(metric_value, width='stretch')
                            else:
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.metric(metric_name.replace('_', ' ').title(), 
                                             value=metric_value if isinstance(metric_value, (int, float)) else str(metric_value))
                    
                    if "predict" in what and returned and returned.get("preds"):
                        st.markdown("### 🔮 Résultats des prédictions")
                        
                        preds_data = returned["preds"]
                        
                        if problem_type == "continuous":
                            st.markdown("""
                            <div class="warning-box">
                                <strong>⚠️ Les prédictions ci-dessous sont des ESTIMATIONS avec une marge d'erreur.</strong><br>
                                Pour une utilisation critique, vérifiez les intervalles de confiance si disponibles.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        try:
                            if isinstance(preds_data, dict):
                                for pred_name, pred_values in preds_data.items():
                                    with st.expander(f"📊 {pred_name.replace('_', ' ').title()}", expanded=False):
                                        if isinstance(pred_values, dict):
                                            df_display = pd.DataFrame({k: [v] if isinstance(v, (str, int, float)) else v 
                                                                      for k, v in pred_values.items()}).T
                                            if pred_name.lower().strip() == "predict":
                                                df_display.columns = ["Classe"]
                                            elif pred_name.lower().strip() == "predict_proba":
                                                df_display.columns = [f"Classe {cl}" for cl in df_display.columns]
                                            st.dataframe(df_display, width='stretch')
                                        else:
                                            st.dataframe(pd.DataFrame(pred_values).T if hasattr(pred_values, '__len__') 
                                                        else pd.DataFrame([pred_values]), width='stretch')
                            else:
                                st.dataframe(pd.DataFrame(preds_data), width='stretch')
                        except Exception as e:
                            st.warning(f"Affichage des prédictions : {str(e)}")
                            st.json(preds_data)

            except Exception as e:
                st.error(f"❌ Erreur lors du chargement des données : {str(e)}")
                st.code(traceback.format_exc(), language="python")

if __name__ == "__main__":
    Interface()()
