#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: isabel
"""

emb= FactorAnalysis(n_components=2)
X1t =emb.fit_transform(X,y)
plt.scatter(X1t[:,0],X1t[:,-1],c=y)
plt.title('Iris dataset FactorAnalysis')
plt.show()

emb= LinearDiscriminantAnalysis(n_components=2)
X2t =emb.fit_transform(X,y)
plt.scatter(X2t[:,0],X2t[:,-1],c=y)
plt.title('Iris dataset LinearDiscriminant')
plt.show()
