#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:50:36 2026

@author: hounsousamuel
"""
import os, sys


CONFIG = {
    "target_name": "target",
    "problem_type": "binary",
    "target_type": "",
    "model_name": "model_binaire.pkl",
    "what": "fit",
    "v": False,
    "opt": False,
    "max_iter":10,
    "dataset_path": "data/dataset_classification.csv",
    "smote": False,
    }

# from sklearn.datasets import load_diabetes, load_iris, load_digits
# import pandas as pd, numpy as np
# X, y = load_diabetes(return_X_y=True)

# frame = pd.DataFrame(X)
# frame.loc[:, 'target'] = y
# frame.to_csv("data/dataset_regression.csv", index=False)
# print(frame.columns)

# X, y = load_iris(return_X_y=True)
# print(np.unique(y))
# frame = pd.DataFrame(X)
# frame.loc[:, 'target'] = y
# frame.to_csv("data/dataset_classification.csv", index=False)
# print(frame.columns)
# X, y = load_digits(return_X_y=True)
# print(np.unique(y))
# frame = pd.DataFrame(X)
# frame.loc[:, 'target'] = y
# frame.to_csv("data/dataset_multi_classe.csv", index=False)
# print(frame.columns)