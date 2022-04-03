import pandas as pd
import numpy as np
import random
from IPython.display import Image
import collections
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from statistics import mode



def euclidean_distance(a,b):
    return np.linalg.norm(a - b)

def create_clusters(X, centroids): 

    labels = []
    for i, x in enumerate(X):
        closest_centroid_idx, _ = min_distance(x, centroids)
        labels.append(closest_centroid_idx)
    return np.array(labels)

def min_distance(sample, centroids):
    distance = [euclidean_distance(sample, centroid) for centroid in centroids]
    distance_min_index = np.argmin(distance)
    return distance_min_index, distance

def select_centroids(X,K):
    
    centroids = []
    random_sample_idx = np.random.choice(X.shape[0], 1, replace=False)
    centroids.append(X[random_sample_idx][0])
    
    #pick next k-1 centroids
    
    for i in range(K-1):
        distances = []
        #find the closest centroid for each data point

        for sample in X:
            temp = []
            for c in centroids:
                distance = euclidean_distance(sample, c)
                temp.append(distance)
            distances.append(min(temp))
           
        #choose the max distance from distances as next centroids 
        next_centroid = X[np.argmax(distances)]
        centroids.append(next_centroid)
    return centroids

def kmeans(X:np.ndarray, K:int, centroids=None, max_iter=30, tolerance=1e-2):
        
    # num_sample, num_features = X.shape
    
    if centroids == 'kmeans++':
        centroids = select_centroids(X, K)
    else:
        #initialize centroids
        random_sample_idx = np.random.choice(X.shape[0], K, replace=False)
        centroids = X[random_sample_idx]
    
    
    for i in range(max_iter):
        #update cluster
        labels = create_clusters(X, centroids)
     
        #update centroids
        prev_centroids = centroids
        
        new_centroids = []
        for i in range(K):
            label_idx_per_cluster = np.where(labels==i)
            new_mean = np.mean(X[label_idx_per_cluster], axis=0)
            new_centroids.append(new_mean)
        centroids = new_centroids

        diff = euclidean_distance(np.array(prev_centroids), np.array(centroids))
        if diff <= tolerance:
            break
    return np.array(centroids), labels


def likely_confusion_matrix(y, labels):
    labels = [1-i for i in labels]
    labels = np.array(labels)
    y_pred = np.zeros(len(y))
    for y_i in np.unique(y):
        mode_ = mode(labels[labels == y_i])
        y_pred[labels == mode_] = y_i

    acc = np.mean(y==y_pred)
    matrix = pd.DataFrame(confusion_matrix(y, y_pred)).rename({0: 'False', 1:'True'}).rename(columns={0: 'False', 1:'True'})
    return acc, y_pred, matrix


## 1D kmean implementation

def initialize_centroids(X, K):
    centroids_dict = collections.defaultdict()
    centroid = np.random.choice(X,K)
    for c in centroid:
        centroids_dict[c] = []
    return centroids_dict

def initiate_clusters(X, centroids):
    clusters_list = list(centroids.keys())
    clusters = centroids
    for x in X:
        distances = []
        for c in centroids.keys():
            distance = np.sqrt(np.sum(np.square(x - c)))
            distances.append(distance)
        min_distance = min(distances)
        cluster_idx = distances.index(min_distance) 
        cluster = clusters_list[cluster_idx]
        clusters[cluster].append(x)
    return clusters

def new_centroids(clusters, X):
    centroids = clusters.keys()
    new_cluster = collections.defaultdict()
    for i, cluster in enumerate(centroids):
        new_mean = np.around(np.mean(clusters[cluster]),2)
        new_cluster[new_mean] = []
    return new_cluster

def reassign(new_cluster, X):
    new_cluster_centroids = list(new_cluster.keys())

    for x in X:
        distances = []
        for c in new_cluster_centroids:
            distance = np.sqrt(np.sum(np.square(x - c)))
            distances.append(distance)
        min_distance = min(distances)
        cluster_idx = distances.index(min_distance) 
        cluster = new_cluster_centroids[cluster_idx]
        new_cluster[cluster].append(x)
    return new_cluster

def find_label(cluster, X):
    clusters = list(cluster.values())
    labels = []
    for x in X:
        for i, each_cluster in enumerate(clusters):
            if x in each_cluster:
                label = list(cluster.keys())[i]
                labels.append(label)
    return labels

def kmeans_1D(X:np.ndarray, K:int, centroids=None, max_iter=30, tolerance=1e-2):
    # initiate the k centroids - default dict
    centroids = initialize_centroids(X, K)
    
    for iter_ in range(max_iter): 
        clusters = initiate_clusters(X, centroids)
        
        prev_centroids = np.array(list(centroids.keys()))
        centroids = reassign(clusters, X)
        current_centroids = np.array(list(centroids.keys()))
        
        diff = np.mean(prev_centroids - current_centroids)
        
        if diff < tolerance:
            labels = find_label(centroids, X)
            centroids = list(centroids.keys())
            return centroids, labels