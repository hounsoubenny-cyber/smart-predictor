#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 15:50:20 2026

@author: hounsousamuel
"""

import os, sys
import pandas as pd
import numpy as np
from classification import BaseModel
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
pd.set_option('display.max_row', 111)
pd.set_option('display.max_columns', 111)


class DataExplain(BaseModel):
    def __init__(self, dataset_path, *args, **kwargs):
        super().__init__(dataset_path=dataset_path)
    
    def _visualise(self, data, target, prefix:str = ""):
        plt.figure(figsize=(24, 12))
        sns.heatmap(data)
        plt.title(f'{prefix}_HEATMAP')
        
        plt.figure(figsize=(24, 12))
        sns.barplot(data)
        plt.title(f"{prefix}_Histogramme")
        
        plt.figure(figsize=(24, 12))
        sns.pairplot(data, kind="reg")
        plt.title(f'{prefix}_PairePlot_line')
        
    def kmeans(self, data, target:str = ""):
        num, not_num = self._get_idx(data[:3])
        transformer = [
            ("num", RobustScaler(), num),
            ("not_num", OneHotEncoder(), not_num)
            ]
        transformer = ColumnTransformer(transformer, n_jobs=-1)
        pip = Pipeline(
            [
                ('transformer', transformer),
                ('kmeans', KMeans(max_iter=500, random_state=self.rs, n_clusters=3))
            ]
            )
        self.model = pip
        self.model.fit(data)
        cluster_center = self.model.named_steps["kmeans"].cluster_centers_
        labels = self.model.named_steps["kmeans"].labels_
        pred = self.model.predict(data)
        print('Labels : ', labels)
        print('Center : ', cluster_center)
        df = pd.DataFrame([{"cluster_labels": labels, "cluster_center": cluster_center}])
        pred = pd.DataFrame(pred)
        return df, pred
        
    def _explain(self, data, dic, prefix=""):
        data = pd.DataFrame(data)
        try:
            describe = data.describe(include="all").fillna(-1)
            dic[f'{prefix}_describe'] = describe
        except:
            pass
        
        try:
            count = data.count()
            dic[f'{prefix}_count'] = count
            self._visualise(count, "target", prefix=f"{prefix}_count")
        except:
            pass
        
        try:
            corr = data.corr()
            dic[f'{prefix}_corr'] = corr
            self._visualise(corr, "target", prefix=f"{prefix}_corr")
        except Exception as e:
            print(e)
        
        try:
            info = data.info()
            dic[f'{prefix}_info'] = info
        except:
            pass
        
        try:
            num_na = data.isna().sum().sum() + data.isnull().sum().sum()
            dic[f'{prefix}_num_na'] = num_na
        except:
            pass
        
        return dic
        
    def explain(self):
        try:
            data = self.load_dataset()
            dic = {}
            if data is not None and not isinstance(data, Exception):
                dic = self._explain(data, dic)
                
            try:
                df, pred = self.kmeans(data)
                print(df, "\n", pred)
            except:
                pass
                
        except Exception as e:
            print('Erreur : ', str(e))
        
        return dic
    
    
if __name__ == "__main__":
    p = "./data/dataset_classification.csv"
    model = DataExplain(p)
    dic = model.explain()
    for k in dic:
        print("=" * 20, k.upper(), "=" * 20)
        print(dic[k])
        print('=' * 50)
