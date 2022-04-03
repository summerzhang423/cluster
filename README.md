# cluster

This report is organized in following topics:
1. What is Kmeans
2. How does it work / Algorithm
3. Visualization with K-means in 1 or 2 dimension(s)
4. Issues with K-means / Improvement
      - kmeans++ 
      - spectral clustering
5. Kmeans accuracty
6. Kmeans application 

## 1. What is Kmeans
K-means is an unsupervised machine learning algorithm that is designed to group similar data points together based on their similarity. Depends on how many groups (or also knowns as clusters) we are trying to define, we want to group our data points into their corresponding clusters as the main objective. "K" here is a generic term indicating there could be however many of clusters we prefer to create.

There are two terms that need to be clarified here, one is "unsupervised" the other is "similarity". K-means is unsupervised model because the input data is not labelled. However, the algorithem will allocate each data point to a cluster. "Similarity" in this context specifically means the euclidean distance between a data point and center of a cluster.

## 2. How does it work / Algorithm
a. Randomly initiate K centroids (pre-determined cluster center) in the same vector space as the data
b. Calculate the distance between each data point and each centroids; find the nearest centroid to the data point and allocate this data point to this cluster
c. After all the data points are assigned to their nearest centroids, compute the mean of the current cluster and update the cluster centroid to the new mean we just calculated
d. Repeat step 2 allowing each data point to find their nearest centroid again
e. Repeat step 2 to 4 until one of the following conditions is met:
      - k centroids have been samples
      - run out the max number of iterations
