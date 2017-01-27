# -*- coding: utf-8 -*-
from enum import Enum
from sklearn.metrics.pairwise import check_pairwise_arrays
import numpy as np

class Metric(Enum):
    #correl = 1
    #chisqr = 2
    intersect = 3
    #bhattacharyya = 4
    #reference for implementation: http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html

def intersect(X, Y = None, dense_output = True):
    X, Y = check_pairwise_arrays(X, Y)
    A = np.array([[sum(np.minimum(x,y)) for y in Y] for x in X])
    #print("Affinity Mat.: ",A)
    #print("shape: ",A.shape)
    return A



class BlockCoder():
    def __init__(self,window_size,clustering_model,metric = Metric.intersect):
        self.window_size = window_size
        self.clustering_model = clustering_model
        self.metric = metric

    def proc(self,input_labels):
        X = self.extract_feature(input_labels)
        if self.metric:
            X = self.make_affinity_mat(X)
        return self.do_clustering(X)

    def extract_feature(self,input_labels):
        r_min = np.min(input_labels)
        r_max = np.max(input_labels)

        X = np.array([self.make_histogram(input_labels[x:x+self.window_size],range(r_min,r_max+2))\
         for x in range(0,len(input_labels),self.window_size)])
        #X = [x/sum(x) for x in X] #normalize
        #print("normed. hist:",X)
        return X

    def make_histogram(self,sub_array,bins):
        hist,bin_edges = np.histogram(sub_array,bins,density=True)
        return hist/sum(hist)

    def make_affinity_mat(self,X):
        if self.metric == Metric.intersect:
            return intersect(X,Y=None)
        return 0 #assert(isinstance(self.metric, Metric), "metric other than intersect has not been implemented yet.")

    def do_clustering(self,X):
        return self.clustering_model.fit_predict(X)

def flatten(arr):
    return [x for y in arr for x in y]


class HyperFeatureCoder():
    def __init__(self, window_sizes, clustering_models, metrics=None):
        self.block_coders = [\
            BlockCoder(ws,cm,m)
            for (ws,cm,m) in zip(window_sizes,clustering_models,metrics)]
        self.depth = len(self.block_coders)

    def window_size_on_orig_data(self,d):
        return np.prod([self.block_coders[i].window_size for i in range(0,d+1)])

    def fit_predict(self,orig_labels):
        labels_tmp = []
        prev_labels = orig_labels
        for (i,block_coder) in enumerate(self.block_coders):
            prev_labels = block_coder.proc(prev_labels)
            labels_tmp.append(prev_labels)

        length = len(orig_labels)
        labels = np.zeros((self.depth+1,length))
        # put labels into matrix shape (care about window_size)
        # put the hyper_feature labels in the same time scale with original labels
        labels = [ flatten( \
                        [ [l]*self.window_size_on_orig_data(i) for l in ls ] \
                       )[0:length] for (i,ls) in enumerate(labels_tmp)]
        return labels
