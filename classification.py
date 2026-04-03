#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:50:10 2026

@author: hounsousamuel
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split as tts, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, HistGradientBoostingRegressor
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, jaccard_score
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from smart_predictor.config import CONFIG
from exception import MyException
_dir_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_dir_, exist_ok=True)


class StepsPipeline:
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **kwargs):
        X = np.asarray(X)
        if y:
            y = np.asarray(y)
        
    
    def transform(self, X, y=None, **kwargs):
        pass
    
    def fit_transform(self, X, y=None, **kwargs):
        pass
    
class BaseModel:
    def __init__(self, dataset_path, model_name="model.pkl"):
        self.model_path = os.path.join(_dir_, "models")
        os.makedirs(self.model_path, exist_ok=True)
        self.model_path = os.path.join(self.model_path, model_name)
        self.dataset_path = str(dataset_path)
        self.rs = 42
        self.v = CONFIG.get('v', False)
        self.lr = 1e-3
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.rs)
        self.history = {
            "models": list(self._get_models().keys()),   # noms des modèles de base utilisés dans le stacking
            "opt": {},      # résultats d'optimisation par modèle {nom: {best_score, best_params, duration}}
            "eval": {}      # résultats d'évaluation {accuracy, classification_report, confusion_matrix, hamming_loss, jaccard_score}
        }
        
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
        xgb = XGBClassifier(
            n_estimators=4000,
            max_depth=7,
            max_leaves=41,
            random_state=self.rs,
            learning_rate=self.lr,
            tree_method="hist",
            n_jobs=-1,
            # verbose=self.v
            )
        dic['xgb'] = ("xgb", xgb)
        
        hist = HistGradientBoostingClassifier(
            max_iter=3000,
            max_leaf_nodes=41,
            learning_rate=self.lr,
            loss='log_loss',
            n_iter_no_change=20,
            early_stopping=True,
            validation_fraction=0.1,
            scoring="f1_macro",
            max_depth=None,
            tol=1e-3,
            random_state=self.rs,
            class_weight="balanced",
            verbose=self.v
            )
        dic['hist'] = ('hist', hist)
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=self.rs,
            verbose=self.v,
            )
        dic['rf'] = ('rf', rf)
        
        log_reg = LogisticRegressionCV(cv=5, tol=1e-5, n_jobs=-1, max_iter=5000)
        pip_log = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ('log_reg', log_reg)
             ]
            )
        dic['log_reg'] = ("log_reg", pip_log)
        return dic
    
    def _get_dict_opt(self, name:str):
        name = str(name).lower()
        PARAMS = {
            "rf": {
                "n_estimators": Integer(500, 1000),
                "max_depth": Categorical([10, 12, None]),
                "max_features": Categorical(['sqrt', "log2"])
                },
            "xgb": {
                "n_estimators": Integer(3000, 8000),
                "max_depth": Categorical([10, 12, 16]),
                "learning_rate": Real(1e-4, 1e-1, prior='log-uniform')
                },
            "hist": {
                "max_iter": Integer(3000, 8000),
                "max_depth": Categorical([10, 12, None]),
                "learning_rate": Real(1e-4, 1e-1, prior='log-uniform'),
                "tol": Real(1e-6, 1e-1, prior='log-uniform')
                }
            }
        return PARAMS.get(name, {})
    
    def _optimize(self, dict_of_models:dict, X, y, max_iter=10):
        exclude = ['log_reg']
        models = [dict_of_models[k] for k in dict_of_models.keys() if k not in exclude]
        best_models = {}
        
        # Renseigner les noms des modèles dans history
        all_names = list(dict_of_models.keys())
        if hasattr(self, 'history'):
            self.history["models"] = all_names

        def _opt():
            for name, model in models:
                s = time.time()
                print('Optimisation de ', str(name).upper())
                search_spaces = self._get_dict_opt(name)
                bayes = BayesSearchCV(
                    model, 
                    search_spaces,
                    scoring="f1_macro",
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

                # Stocker les résultats d'optimisation dans history
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
        self.history["models"] = list(models.keys())
        
        log_reg = LogisticRegressionCV(cv=5, tol=1e-5)
        meta = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ('log_reg', log_reg)
             ]
            )
        if not opt:
            stack = StackingClassifier(
                list(models.values()),
                final_estimator=meta,
                passthrough=True,
                n_jobs=-1,
                cv=self.cv,
                stack_method="predict_proba"
                )
        else:
            if not X is None and not y is None:
                X = np.asarray(X)
                y = np.asarray(y)
                best_models = self._optimize(models, X, y, max_iter)
                stack = StackingClassifier(
                    list(best_models.values()),
                    final_estimator=meta,
                    passthrough=True,
                    n_jobs=-1,
                    cv=self.cv,
                    stack_method="predict_proba"
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
    
class ModelBinaire(BaseModel):
    def __init__(self, dataset_path, model_name="model_binaire.pkl"):
        model_name = model_name or "model_binaire.pkl"
        super().__init__(dataset_path, model_name)
        self.target_encoder = LabelEncoder()
        self.model = None
        self.imputer = IterativeImputer(
            estimator=HistGradientBoostingRegressor(
                max_iter=1000,
                n_iter_no_change=30
                ),
            max_iter=20,
            tol=1e-5,
            )
    
    def _load_model(self):
        data = self.load_model()
        if data and  isinstance(data, dict) :
            self.model = data['model']
            self.target_encoder = data["target_encoder"]
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
    
    def fit(self, target_name:str, smote:bool = False, opt:bool = True, max_iter:int = 10, average="binary"):
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
                
                y = self.target_encoder.fit_transform(y)
                if int(X.isna().sum().sum()) != 0:
                    X = self._impute(X_numpy)
                if smote:
                    try:
                        X, y = SMOTE(k_neighbors=20, random_state=self.rs).fit_resample(X, y)
                    except Exception as e:
                        print('Erreur SMOTE : ', str(e))
                        print("SMOTE ignoré !")
                
                X_numpy = X.to_numpy()
                X_train, X_test, y_train, y_test = tts(X_numpy, y, test_size=0.1)
                model = self._model(X, y, opt, max_iter)
                model.fit(X_train, y_train)
                self.model = model
                self.f = True
                self.evaluate(X_test, y_test, average)
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
                        "target_encoder": self.target_encoder, 
                        "imputer": self.imputer
                    }
                )
            return self
    
    def evaluate(self, X, y, average="binary"):
        print()
        print("="*20, "EVALUATION", "="*20)
        predict = self.model.predict(X)
        metrics = {}

        try:
            cr = classification_report(y, predict, output_dict=True)
            print('Classification report : \n', classification_report(y, predict))
            metrics["classification_report"] = pd.DataFrame(cr).T
        except Exception as e:
            print('Erreur classification_report : ', e)

        try:
            cm = confusion_matrix(y, predict)
            print('Confusion matrix : \n', cm)
            classes = self.target_encoder.classes_
            metrics["confusion_matrix"] = pd.DataFrame(cm, index=[f"Réel_{str(k)}" for k in classes], columns=[f"Predit_{str(k)}" for k in classes])
        except Exception as e:
            print('Erreur confusion_matrix : ', e)

        try:
            score = self.model.score(X, y)
            print('Score : ', score)
            metrics["accuracy"] = round(float(score), 4)
        except Exception as e:
            print('Erreur score : ', e)

        try:
            hml = hamming_loss(y, predict)
            print('hamming_loss(plus c\'est petit mieux c\'est) : ', hml)
            metrics["hamming_loss"] = round(float(hml), 4)
        except Exception as e:
            print('Erreur hamming_loss : ', e)

        try:
            js = jaccard_score(y, predict, average=average)
            print('Jaccard score(plus c\'est grand mieux c\'est) : ', js)
            metrics["jaccard_score"] = round(float(js), 4)
        except Exception as e:
            print('Erreur jaccard_score : ', e)

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
        y_pred = np.asarray(self.model.predict(X)).astype(int)
        y_pred_proba = np.asarray(self.model.predict_proba(X)).astype(float)
        classes_ = self.target_encoder.classes_
        try:
            classes_ = [int(x) for x in classes_]
        except:
            pass
        
        to_return = {
            "predict": {
                int(i): int(pred) if isinstance(pred, (np.number, int, float)) else pred 
                for i, pred in enumerate(self.target_encoder.inverse_transform(y_pred))
            },
            "predict_proba": {
                int(i): {str(k): float(v) for k, v in dict(zip(classes_, [float(x) for x in line])).items()}  
                for i, line in enumerate(y_pred_proba)
            },
        }
        return to_return

class ModelMultiClass(ModelBinaire):
    def __init__(self, dataset_path, model_name="model_multi_class.pkl"):
        model_name = model_name or "model_multi_class.pkl"
        super().__init__(dataset_path, model_name)
        self.target_encoder = LabelEncoder()
        self.model = None
        self.imputer = IterativeImputer(
            estimator=HistGradientBoostingClassifier(
                max_iter=1000,
                n_iter_no_change=30
                ),
            max_iter=20,
            tol=1e-5,
            )
    def fit_multiclass(self, target_name:str, smote:bool = False, opt:bool = True, max_iter:int = 10, average="weighted"):
        return super().fit(target_name, smote, opt, max_iter, average)
        
class ModelMultiLabel(ModelBinaire):
    def __init__(self, dataset_path, model_name="model_multi_label.pkl"):
        model_name = model_name or "model_multi_label.pkl"
        super().__init__(dataset_path, model_name)
        self.target_encoder = MultiLabelBinarizer()
        self.model = None
        self.imputer = IterativeImputer(
            estimator=HistGradientBoostingClassifier(
                max_iter=1000,
                n_iter_no_change=30
                ),
            max_iter=20,
            tol=1e-5,
            )
        
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
            ('stacking', OneVsRestClassifier(stack, n_jobs=-1, verbose=self.v))
            ]
        model = Pipeline(
            steps=steps,
            verbose=self.v
            )
        return model
    
    def fit_multilabel(self, target_name:str, smote:bool = False, opt:bool = True, max_iter:int = 10, average="samples"):
        return super().fit(target_name, smote, opt, max_iter, average)
    
if __name__ == "__main__":
    import json, pprint
    p_ = "/home/hounsousamuel/PROJETS/smart_predictor/data/dataset_classification.csv"
    # p_ = "/home/hounsousamuel/PROJETS/smart_predictor/data/dataset_multi_classe.csv"
    # m = ModelBinaire(p_)
    data = pd.DataFrame(pd.read_csv(p_))
    print(len(data), "données !")
    X = data.drop("target", axis=1, inplace=False)
    y = data['target']
    print(np.unique_values(y))
    to_pred = X[:10].to_numpy()
    i = 0
    m = ModelMultiClass(p_)
    print(m._get_idx(X))
    input()
    m.fit(target_name="target", opt=True, max_iter=1)
    
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
    
    print(m.history)