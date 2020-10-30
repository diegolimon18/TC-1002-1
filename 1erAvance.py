#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:21:06 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis

sp=pd.read_csv('/Users/diegovelazquez/Downloads/iris.csv')
sp['Tipo_Flor']=sp['Tipo_Flor'].replace(['Iris-versicolor','Iris-virginica','Iris-setosa'],[0,1,2])
data=sp.values
X= data[:,0:-1]
y=data[:,-1]
