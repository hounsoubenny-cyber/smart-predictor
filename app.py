#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:56:35 2026

@author: hounsousamuel
"""

import os, sys
import streamlit as st
import joblib
import pandas as pd
import tempfile
from main import main

class Interface:
    def __init__(self):
        self.th = None
        
    def __call__(self):
        if "result" not in st.session_state:
            st.session_state.result = None
        if "model" not in st.session_state:
            st.session_state.model = None

    
        st.set_page_config(
            page_title="SmartPredictor"
            )
        st.title("🤖 SmartPredictor - by @hounsoubenny-cyber")
        st.caption("Auto-ML · Détecte ton problème et entraîne le bon modèle")
        with st.sidebar:
            st.header("Configuration")
            st.divider()
            optimize = st.sidebar.toggle("Bayesian Optimization", value=True, key="1")
            with st.container(border=True):
                max_iter = 1
                if optimize:
                    max_iter = st.number_input("Nombre maximum d'itération pour optimisation :", max_value=200, min_value=1, key="2", value=10)
                what = st.selectbox("Action à faire :", options=["fit_predict", "predict", "fit"], key="3")
                problem_type = st.selectbox(
                    "Type du problème : ", 
                    options=["binary", 'continuous', 'multiclass', 'multilabel', "explain", "deviner"],
                    key="4"
                    )
                # target_name = st.text_input("Nom de la target dans les données", key="5")
                smote = False
                if what == "fit":
                    smote = st.selectbox("Appliquer smote", options=["True", "False"]) == "True"
                    
                model_path = st.text_input("Chemin de sauvegarde du modèle", value="model.pkl", key="5").strip()
                
                verbose = st.selectbox("Verbosité : ", options=["True", "False"], key="6") == "True"
            
        file = st.file_uploader("Uploader vos données : ", type=["csv", "json", "pkl", "joblib", "xls"], key="7", accept_multiple_files=False)
        # bar = st.progress(value=0, text="Preogression")
        # with st.spinner(show_time=True, text="Progression"):
        #     import time
        #     for i in range(100):
        #         time.sleep(1)
        #         bar.progress(i + 1)
        if file:
            ext = os.path.splitext(file.name)[-1].lower()
            func = pd.read_csv
            if ext == ".pkl":
                func = pd.read_pickle
            elif ext == ".joblib":
                func = joblib.load
            elif ext == ".json":
                func = pd.read_json
            elif ext == ".xls":
                func = pd.read_excel
                
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
            apercu_size = st.number_input("Taille de l'aperçu", min_value=1, max_value=len(data) - 1, key="8", value=min(10, len(data)))
            st.write(f"Aperçu ({apercu_size} première ligne)")
            st.dataframe(data.head(apercu_size))
            
            target_name = st.selectbox("Nom de la target : ", options=list(data.columns), key="9")
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
            
            if st.button("Démarrer !", type="secondary", icon=":material/mood:", width="stretch", key="10"):
                with st.spinner(text="En cours...", show_time=True):
                    st.warning("Ne plus rien modifier, démarrage.")
                    st.write("Config : ")
                    st.dataframe(pd.DataFrame({k : [v] for k, v in CONFIG.items()}))
                    if CONFIG["what"] != "fit_predict":
                        returned = main(CONFIG, include_model=True)
                        model = returned["model"]
                    else:
                        CONF_COPY = CONFIG.copy()
                        CONF_COPY["what"] = "fit"
                        returned = main(CONF_COPY, include_model=True)
                        CONF_COPY["what"] = "predict"
                        st.info("Fit terminé, début predictions !")
                        model = returned["model"]
                        
                        returned = main(CONF_COPY, include_model=True)
                        
                    st.session_state.model = model
                    st.session_state.result = returned
                    st.session_state.running = False
                                
            
            if st.session_state.result:
                returned = st.session_state.result
                model = st.session_state.model
                
                if returned.get("errors", []):
                    st.write("Erreurs survenue : \n")
                    for err in returned["errors"]:
                        st.error(err)
                else:                    
                    if what == "fit":
                        st.success(f"Entrainement réussi, model dans {returned['model'].model_path}")
                    elif what == "predict":
                        st.success("Prediction réussi")
                    else:
                        st.success(f"Entrainement réussi, model dans {returned['model'].model_path}")
                        st.success("Prediction réussi")
                    
                print(model.history)
                st.divider()
                st.header("Models entrainés")
                st.info(', '.join(model.history['models']))
                
                if model.history.get("opt", {}):
                    st.header("Résultats des optimisations bayésiens")
                    for k, v in model.history.get("opt", {}).items():
                        st.divider()
                        st.subheader(f"Model {str(k).title()}")
                        for i, j in v.items():
                            if isinstance(j, pd.DataFrame):
                                st.subheader(str(i).capitalize())
                                st.dataframe(j)
                            else:
                                st.subheader(str(i).capitalize())
                                st.metric(i, value=j)
                    st.divider()
                
                else:
                    st.divider()
                    
                if model.history.get("eval", {}):
                    st.header("Résultat d'évalutions")
                    st.divider()
                    for k, v in model.history.get("eval", {}).items():
                        if isinstance(v, pd.DataFrame):
                            st.subheader(str(k).capitalize())
                            st.dataframe(v)
                        else:
                            st.subheader(str(k).capitalize())
                            st.metric(k, value=v)
                    st.divider()
                
                else:
                    st.divider()
                    
                if "predict" in what:
                    st.header("Prédiction : ")
                    # for k, v in returned["preds"].items():
                    #     if v:
                    #         st.write(k)
                    #         print(v)
                    #         st.dataframe(pd.DataFrame((v)))
                    #         st.divider()
                    st.dataframe(pd.DataFrame(returned["preds"]))
                    for k, v in returned["preds"].items():
                        st.subheader(str(k).capitalize())
                        if "predict" in k:
                            st.dataframe(pd.DataFrame({i: [j] if isinstance(j, (str, int, float)) else j for i, j in v.items()}).T)
                        else:
                            st.dataframe(pd.DataFrame(v).T)
            
        
if __name__ == "__main__":
    Interface()()
                
