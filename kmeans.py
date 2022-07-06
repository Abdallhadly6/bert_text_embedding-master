import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = np.array([[1.0,1.0],
    [1.5,2.5],
    [3.0,4.0],
    [5.0,7.0],
    [3.5,5.0],
    [4.5,5.0],
    [3.5,4.0],])


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
print(cluster.labels_)
