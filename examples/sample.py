# -*- coding: utf-8 -*-

import sys
import numpy as np
import hyper_feature_coding as hfc
from sklearn.cluster import SpectralClustering
from sklearn.datasets import load_iris
from sklearn import cluster
import sklearn.metrics as sm


if __name__ == "__main__":

    print("prepare clustering models in each depth level.")
    n_clusters = [2]#,8,4]
    cmodels = [SpectralClustering(n_clusters=num, affinity='precomputed', assign_labels='discretize') for num in n_clusters]

    print("set window_size in each depth level")
    # 15/30fps=0.5s, 2.5s, 12.5s
    window_sizes = [1]#,5,5]
    metrics = [hfc.Metric.intersect]# * 3

    hfcoder = hfc.HyperFeatureCoder(window_sizes,cmodels,metrics)

    print("load motion feature sequence (dummy)")
    X = np.array(load_iris().data)

    print("clustering raw feature sequence")
    sc0 = SpectralClustering(n_clusters=32, affinity= 'cosine', assign_labels='discretize')
    labels = sc0.fit_predict(X)

    print("hyper_feature_coding")
    layered_labels = hfcoder.fit_predict(labels)
    layered_labels.insert(0, labels)

    np.savetxt(sys.stdout.buffer, np.array(layered_labels).T, fmt='%d', delimiter=',')
