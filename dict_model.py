#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 17:33:42 2026

@author: hounsousamuel
"""

from classification import ModelBinaire, ModelMultiClass, ModelMultiLabel
from regression import ModelRegression
from data_explain import DataExplain

DICT_MODELS = {
    "continuous": lambda x, y = None: ModelRegression(x, y),
    "binary": lambda x, y = None: ModelBinaire(x, y),
    "multiclass": lambda x, y = None: ModelMultiClass(x, y),
    "multilabel": lambda x, y = None: ModelMultiLabel(x, y),
    "explain": lambda x, y = None: DataExplain(x),
    }