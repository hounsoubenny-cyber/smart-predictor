#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:49:58 2026

@author: hounsousamuel
"""
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split as tts, RepeatedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from exception import MyException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import CONFIG

import warnings
warnings.filterwarnings('ignore')

_dir_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_dir_, exist_ok=True)

class BaseModel:
    def __init__(self, dataset_path, model_name="model.pkl"):
        model_name = model_name or "model_regression.pkl"
        self.model_path = os.path.join(_dir_, "models")
        os.makedirs(self.model_path, exist_ok=True)
        self.model_path = os.path.join(self.model_path, model_name)
        self.dataset_path = str(dataset_path)
        self.rs = 42
        self.v = CONFIG.get('v', False)
        self.lr = 1e-3
        self.cv =3
        self.marge = None
    
    def save_model(self, to_save):
        try:
            joblib.dump(to_save, self.model_path, compress=5)
            print(f'Modèle sauvegardé avec succès dans {self.model_path}')
        except Exception as e:
            print(f'Erreur sauvegarde du modèle dans {self.model_path} : {str(e)}')
    
    def load_model(self):
        try:
            data = joblib.load(self.model_path)
            print(f'Modèle chargé avec succès, type de la donné chargé : {type(data)}')
            print('Chargement effectué avec succès depuis : ', self.model_path)
            return data
        except Exception as e:
            print(f'Erreur de chargement du modèle depuis {self.model_path} : {str(e)}')
            return e
    
    def _get_good_func(self, path:str):
        if path.endswith(('.pkl', ".joblib")):
            return joblib.load
        
        elif path.endswith('.csv'):
            return pd.read_csv
        
        elif path.endswith('.json'):
            return pd.read_json
        
        elif path.endswith('.xls'):
            return pd.read_excel
        
        return joblib.load
    
    def load_dataset(self):
        func = self._get_good_func(self.dataset_path)
        try:
            data = func(self.dataset_path)
            print('Données chargées avec succès depuis : ', self.dataset_path)
            return data
        except Exception as e:
            print(f'Erreur de chargement du modèle depuis {self.dataset_path} : {str(e)}')
            return e
    
    def _impute(self, X):
        X = self.imputer.fit_transform(X)
        return np.asarray(X)
    
    def _get_models(self):
        dic = {}
        xgb = XGBRegressor(
            n_estimators=5000,
            max_depth=14,
            max_leaves=50,
            random_state=self.rs,
            learning_rate=self.lr,
            tree_method="hist",
            n_jobs=-1,
            )
        dic['xgb'] = ("xgb", xgb)
        
        lgb = LGBMRegressor(
            n_estimators=5000,
            n_jobs=-1,
            max_depth=16,
            num_leaves=50,
            learning_rate=self.lr,
            verbose=-1
            )
        dic['lgb'] = ('lgb', lgb)
        
        hist = HistGradientBoostingRegressor(
            max_iter=3000,
            max_leaf_nodes=50,
            learning_rate=self.lr,
            n_iter_no_change=100,
            early_stopping=True,
            validation_fraction=0.1,
            max_depth=None,
            tol=1e-4,
            random_state=self.rs,
            verbose=self.v
            )
        dic['hist'] = ('hist', hist)
        
        rf = RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            n_jobs=-1,
            random_state=self.rs,
            verbose=self.v,
            )
        dic['rf'] = ('rf', rf)
        
        # linear = LinearRegression(
        #     n_jobs=-1
        #     )
        # dic['linear'] = ("linear", linear)
        
        # ridge_cv = RidgeCV(
        #     cv=self.cv,
        #     alphas=np.linspace(1e-4, 100, 50)
        #     )
        # dic["ridge_cv"] = ('ridge_cv', ridge_cv)
        
        return dic
    
    def _get_dict_opt(self, name:str):
        name = str(name).lower()
        PARAMS = {
            "rf": {
                "n_estimators": Integer(500, 1500),
                "max_depth": Categorical([4, 6, 8, 10, 12, None]),
                "max_features": Categorical(['sqrt', "log2"])
                },
            "xgb": {
                "n_estimators": Integer(3000, 8000),
                "max_depth": Categorical([10, 12, 16, 18]),
                "learning_rate": Real(1e-4, 1e-1, prior='log-uniform')
                },
            "hist": {
                "max_iter": Integer(3000, 8000),
                "max_depth": Categorical([10, 12, 14, None]),
                "learning_rate": Real(1e-4, 1e-1, prior='log-uniform'),
                "tol": Real(1e-6, 1e-1, prior='log-uniform')
                },
            "lgb": {
                "n_estimators": Integer(3000, 8000),
                "max_depth": Categorical([10, 12, 16, 18]),
                "learning_rate": Real(1e-4, 1e-1, prior='log-uniform')
                }
            }
        return PARAMS.get(name, {})
    
    def _optimize(self, dict_of_models:dict, X, y, max_iter=10):
        exclude = ['linear', "ridge_cv"]
        models = [dict_of_models[k] for k in dict_of_models.keys() if k not in exclude]
        best_models = {}

        # Renseigner les noms des modèles dans history
        if hasattr(self, 'history'):
            self.history["models"] = list(dict_of_models.keys())

        def _opt():
            for name, model in models:
                s = time.time()
                print('Optimisation de ', str(name).upper())
                search_spaces = self._get_dict_opt(name)
                bayes = BayesSearchCV(
                    model, 
                    search_spaces,
                    n_iter=max_iter,
                    n_jobs=-1,
                    cv=self.cv,
                    return_train_score=True
                    )
                bayes.fit(X, y)
                best = bayes.best_estimator_
                best_models[name] = (name, best)
                duration = round(time.time() - s, 2)
                print("Fit de ", str(name).upper(), " fini : \n")
                print('Meilleur score cv : ', bayes.best_score_)
                print('Meillsur params : ', dict(bayes.best_params_))
                print('Infos d\'optimisations : \n', pd.DataFrame(bayes.cv_results_))
                print(f"Fit fini en : {duration} secondes !")
                print()

                # Stocker dans history
                if hasattr(self, 'history'):
                    self.history["opt"][name] = {
                        "best_score_cv": round(float(bayes.best_score_), 4),
                        "best_params":   pd.DataFrame({str(k): [v] if isinstance(v, (int, float, str, np.number)) else v for k, v in dict(bayes.best_params_).items()}),
                        "duration_sec":  duration,
                        "cv_results":    pd.DataFrame(bayes.cv_results_)
                    }
        _opt()
                
        for name in exclude:
            if name in dict_of_models:
                _, model = dict_of_models[name]
                best_models[name] = (name, model)
                
        return best_models
        
    
    def _stack(self, opt=False, X=None, y=None, max_iter=10):
        models = self._get_models()

        # Toujours renseigner les noms des modèles dans history
        if hasattr(self, 'history'):
            self.history["models"] = list(models.keys())

        meta  = RidgeCV(
            cv=self.cv,
            alphas=np.linspace(1e-4, 100, 50)
            )
        if not opt:
            stack = StackingRegressor(
                list(models.values()),
                final_estimator=meta,
                passthrough=True,
                n_jobs=-1,
                cv=self.cv,
                verbose=self.v,
                )
        else:
            if not X is None and not y is None:
                X = np.asarray(X)
                y = np.asarray(y)
                best_models = self._optimize(models, X, y, max_iter)
                stack = StackingRegressor(
                    list(best_models.values()),
                    final_estimator=meta,
                    passthrough=True,
                    n_jobs=-1,
                    cv=self.cv,
                    verbose=self.v,
                )
        return stack
    
    def _get_idx(self, X):
        X = np.asarray(X)
        number, not_number = [], []
        for i in range(X.shape[1]):
            if any(isinstance(x, (np.number, int, float)) for x in X[:, i]):
                if not i in number:
                    number.append(i)
            else:
                if not i in not_number:
                    not_number.append(i)
            
        return number, not_number
    
class ModelRegression(BaseModel):
    def __init__(self, dataset_path, model_name="model_regression.pkl"):
        super().__init__(dataset_path, model_name)
        self.model = None
        self.imputer = IterativeImputer(
            estimator=HistGradientBoostingRegressor(
                max_iter=1000,
                n_iter_no_change=30
                ),
            max_iter=20,
            tol=1e-5,
            )
        self.history = {
            "models": list(self._get_models().keys()),   # noms des modèles de base utilisés dans le stacking
            "opt": {},      # résultats d'optimisation par modèle {nom: {best_score, best_params, duration}}
            "eval": {}      # résultats d'évaluation {mae, mse, r2, score}
        }
    
    def _load_model(self):
        data = self.load_model()
        if data and  isinstance(data, dict) :
            self.model = data['model']
            self.imputer = data['imputer']
    
    def _model(self, X, y, opt=True, max_iter=10):
        X_ = X[:3] # Pour réduire les itérations
        num, not_num = self._get_idx(X_)
        transformer = [
            ("num", RobustScaler(), num),
            ("not_num", OneHotEncoder(), not_num)
            ]
        transformer = ColumnTransformer(
            transformers=transformer,
            n_jobs=-1
            )
        stack = self._stack(opt, X, y, max_iter)
        steps = [
            ('transformer', transformer),
            ('stacking', stack)
            ]
        model = Pipeline(
            steps=steps,
            verbose=self.v
            )
        return model
    
    def fit(self, target_name:str, smote:bool = False, opt:bool = True, max_iter:int = 10):
        try:
            self.f = False
            if not target_name:
                target_name = CONFIG.get('target_name', "")
            data = self.load_dataset()
            if data is not None and not isinstance(data, Exception):
                frame = pd.DataFrame(data)
                if not target_name in frame.columns:
                    exc = MyException("Target Name absente dans les données!")
                    exc.txt = "Target Name absente dans les données!"
                    raise exc
                X = frame.drop(target_name, inplace=False, axis=1)
                y = frame[target_name]
                if len(X.shape) != 2:
                    exc = MyException("X n'est pas de dimension 2")
                    exc.txt = "X n'est pas de dimension 2"
                    raise exc
                    
                X_numpy = X.to_numpy()
                for i in range(X_numpy.shape[1]):
                    first_type = type(X_numpy[:, i][0])
                    if any(not isinstance(x, first_type) for x in X_numpy[:, i]):
                        exc = MyException("Dans une colonne, tout les éléments doivent être de même type !")
                        exc.txt = "Dans une colonne, tout les éléments doivent être de même type !"
                        exc.col = i
                        print("L'erreur est dans la colonne ", i + 1)
                        raise exc
                    
                    
                if int(X.isna().sum().sum()) != 0:
                    X = self._impute(X_numpy)
                
                if smote:
                    pass
                
                X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1)
                model = self._model(X, y, opt, max_iter)
                model.fit(X_train, y_train)
                self.model = model
                self.f = True
                self.evaluate(X_test, y_test)
            else:
                exc = MyException("Target Name non fournie !")
                exc.txt = "Target Name non fournie !"
                raise exc
                
        except Exception as e:
            print('Erreur lors du fit : ', str(e))
            import traceback
            traceback.print_exc()
            
        finally:
            if self.f:
                self.save_model(
                    {
                        "model": model, 
                        "imputer": self.imputer
                    }
                )
            return self
    
    def _get_order(self, num):
        order = len(str(int(num)))
        return int("1" + "0"*(order - 1))
        
    def evaluate(self, X, y):
        print()
        print("="*20, "EVALUATION", "="*20)
        predict = np.asarray(self.model.predict(X))
        metrics = {}

        try:
            mae = mean_absolute_error(y, predict) / self._get_order(np.max(np.abs(y - predict)))
            print('mean_absolute_error(plus c\'est proche de 0 mieux c\'est) : ', mae)
            metrics["mae"] = round(float(mae), 4)
        except Exception as e:
            print('Erreur mae : ', e)

        try:
            mse = mean_squared_error(y, predict) / self._get_order(np.max((y - predict) ** 2))
            print('mean_squared_error(plus c\'est proche de 0 mieux c\'est) : ', mse)
            metrics["mse"] = round(float(mse), 4)
        except Exception as e:
            print('Erreur mse : ', e)

        try:
            self.marge = min(metrics.get("mae", 0), metrics.get("mse", 0))
        except:
            pass

        try:
            r2 = r2_score(y, predict)
            print('r2_score(plus c\'est proche de 1 mieux c\'est) : ', r2)
            metrics["r2"] = round(float(r2), 4)
        except Exception as e:
            print('Erreur r2 : ', e)

        try:
            score = self.model.score(X, y)
            print('Score : ', score)
            metrics["score"] = round(float(score), 4)
        except Exception as e:
            print('Erreur score : ', e)

        print("="*20, "FIN", "="*20)
        print()

        # Stocker dans history
        if hasattr(self, 'history'):
            self.history["eval"] = metrics

        return metrics
    
    def predict(self, X):
        if not self.model:
            self._load_model()
            if not self.model:
                raise ValueError('Veuillez d\'abord entrainé le model avec la méthode fit !')
        X = np.asarray(X)
        y_pred = np.asarray(self.model.predict(X)).astype(float)
        to_return = {
            "predict": {int(i):float(pred) for i, pred in enumerate(y_pred)},
            "marge_erreur": float(f"{self.marge * 100:.2f}") if self.marge else "Non calculé !"
            }
        return to_return
              
if __name__ == "__main__":
    import json, pprint
    p_ = "/home/hounsousamuel/PROJETS/smart_predictor/data/dataset_regression.csv"
    m = ModelRegression(p_)
    m.fit(target_name="target", opt=True, max_iter=1)
    print("Fitted : ", m.f)
    data = pd.DataFrame(pd.read_csv(p_))
    X = data.drop("target", axis=1, inplace=False)
    y = data['target']
    to_pred = X[:10].to_numpy()
    i = 0
    for line in to_pred:
        line = line.reshape(1, -1)
        print()
        print(line)
        try:
            print(json.dumps(m.predict(line), indent=2, ensure_ascii=False))
        except:
            print(pprint.pprint(m.predict(line), indent=2))
        print(y[i])
        i += 1
        print()