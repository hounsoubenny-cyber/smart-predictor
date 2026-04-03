#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:50:28 2026

@author: hounsousamuel
"""

import os, sys
import json, pprint
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from config import CONFIG
from dict_model import DICT_MODELS
from data_explain import DataExplain
from classification import MyException, ModelBinaire, ModelMultiClass, ModelMultiLabel, BaseModel
from regression import ModelRegression
from exception import explain_exc

_base_model = None
REQUIRED_KEYS = ["problem_type", "dataset_path", "what"]
REQUIRED_VALUES = ['dataset_path']

def _validate_config():
    if not CONFIG:
        exc = MyException("Config vide !")
        exc.txt = "Config vide !"
        raise exc
    
    for k in REQUIRED_KEYS:
        if not k in CONFIG:
            exc = MyException(f'Clé absente ({k})')
            exc.txt = f'Clé absente ({k})'
            raise exc
        else:
            if k in REQUIRED_VALUES:
                val = CONFIG[k]
                if not val:
                    exc = MyException(f"Valeur vide pour la clé : {k}")
                    exc.txt = f"Valeur vide pour la clé : {k}"
                    raise exc
            if k == "what":
                if CONFIG[k] == "predict":
                    if not 'model_name' in CONFIG:
                        exc = MyException("clé what a pour valeur prédict mais pas de model_path")
                        exc.txt = "clé what a pour valeur prédict mais pas de model_path"
                        raise exc
                
    return True

def _presente_preds(preds:dict):
    try:
        preds = {k:v for k, v in preds.items() if k.lower() != "model"}
        to_return = json.dumps(preds,indent=2, ensure_ascii=False)
        print(to_return)
    except:
        to_return = pprint.pprint(preds, indent=2)
        print(to_return)

def _action(problem_type:str, target_name:str, dataset_path:str, model_name:str=None, smote:bool = False):
    if problem_type not in ('continuous', 'binary', 'multiclass', 'multilabel', "explain"):
        exc = MyException("problem_type inconnue")
        exc.txt = "problem_type inconnue"
        raise exc
        
    else:
        model = DICT_MODELS[problem_type](x=dataset_path, y=model_name)
        to_return = {"model": model, "preds": None, "errors": []}
        if problem_type in ('continuous', 'binary', 'multiclass', 'multilabel'):
            what = str(CONFIG['what']).strip().lower()
            if what not in ('predict', "fit"):
                exc = MyException("what non reconnue !")
                exc.txt = "what non reconnue !"
                raise exc
                
            else:
                if what == "fit":
                    try:
                        print("Entrainement du model")
                        model.fit(
                            opt=CONFIG.get('opt', False),
                            max_iter=CONFIG.get('max_iter', 10),
                            target_name=target_name,
                            smote=smote
                        )
                        
                        to_return["model"] = model
                    except Exception as e:
                        explain_exc(e)
                        to_return['errors'].append(str(e))
                        
                else:
                    try:
                        data = model.load_dataset()
                        if data is not None and not isinstance(data, Exception):
                            data = pd.DataFrame(data)
                            if target_name in list(data.columns):
                                X = data.drop(target_name, axis=1, inplace=False)
                                
                            else:
                                X = data.copy(deep=True)
                            model.load_model()
                            preds = model.predict(X)
                            to_return['preds'] = preds
                            to_return["model"] = model
                            preds = _presente_preds(preds)
                        else:
                            print('Erreur de chargement de vos données, veuillez vérifier chemins et si les données sont valides')
                    except Exception as e:
                        explain_exc(e)
                        to_return['errors'].append(str(e))
        else:
            pass
        return to_return
            
def main(conf = None, include_model:bool = False):
    global _base_model
    global CONFIG
    returned = {}
    if conf:
        CONFIG = conf
        print(CONFIG)
    try:
        is_valide = _validate_config()
        if is_valide and not isinstance(is_valide, Exception):
            _base_model = BaseModel(CONFIG['dataset_path'])
            data = _base_model.load_dataset()
            if not data is None and not isinstance(data, Exception):
                problem_type = CONFIG['problem_type']
                target_name = CONFIG['target_name']
                data = pd.DataFrame(data)
                cols = list(data.columns)
                if not target_name in cols:
                    problem_type = "explain"
                else:
                    if problem_type:
                        if not problem_type.strip().lower() in ('continuous', 'binary', 'multiclass', 'multilabel'):
                            print('problem_type invalide, veuillez consulter la documentation.')
                            print("Fallback, essai de devination de problem_type")
                            try:
                                problem_type = type_of_target(np.array(data[target_name]))
                                for a in ('binary', "multiclass", "multilabel", "continous"):
                                    if a in problem_type:
                                        problem_type = a
                                        break
                                    
                            except Exception as e:
                                print('Erreur lors de la devination du problem_type. Veuillez le precisez vous même.')
                                print('Erreur: ', str(e))
                                problem_type = 'explain'
                    else:
                        print('problem_type non fournie, veuillez consulter la documentation.')
                        print("Fallback, essai de devination de problem_type")
                        try:
                            problem_type = type_of_target(np.array(data[target_name]))
                            for a in ('binary', "multiclass", "multilabel", "continous"):
                                if a in problem_type:
                                    problem_type = a
                                    break
                                
                        except Exception as e:
                            print('Erreur lors de la devination du problem_type. Veuillez le precisez vous même.')
                            print('Erreur: ', str(e))
                            problem_type = 'explain'
                print('Problem_type : ', problem_type)
                model_name = CONFIG.get('model_name', None) or None
                smote = CONFIG.get("smote", False)
                returned = _action(
                    problem_type=problem_type,
                    target_name=target_name,
                    dataset_path=CONFIG["dataset_path"],
                    model_name=model_name,
                    smote=smote
                )
                returned_ = {k:v for k, v in returned.items() if k.lower() != "model"} if not include_model else returned
                returned_['probleme_type'] = problem_type
                _presente_preds(returned_)
            else:
                print('Erreur lors du chargement de votre dataset, veuillez verifier le chemin et vos données.')
                print("Aussi il est conseillé d'utiliser des fichiers d'extensions '.pkl', '.joblib', '.csv', '.json' .")
            return returned_
                    
                
    except Exception as e:
        explain_exc(e)
        return returned

if __name__ == "__main__":
    returned = main()