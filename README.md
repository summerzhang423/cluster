# cluster

This report is organized in following topics:
[] What is Kmeans
[] How does it work / Algorithm
[] Visualization with K-means in 1 or 2 dimension(s)
[] Issues with K-means / Improvement
      - kmeans++ 
      - spectral clustering
[] Kmeans accuracty
[] Kmeans application

1. What is Kmeans
K-means is an unsupervised machine learning algorithm that is designed to group similar data points together based on their similarity. Depends on how many groups (or also knowns as clusters) we are trying to define, we want to group our data points into their corresponding clusters as the main objective. "K" here is a generic term indicating there could be however many of clusters we prefer to create.

There are two terms that need to be clarified here, one is "unsupervised" the other is "similarity". K-means is unsupervised model because the input data is not labelled. However, the algorithem will allocate each data point to a cluster. "Similarity" in this context specifically means the euclidean distance between a data point and center of a cluster.
