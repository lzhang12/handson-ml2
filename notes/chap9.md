# Chapter 9

## Unsupervised Learning

> If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake. 
>
> ​																																				--- Yann LeCun

* Clustering

  * Applications

    * customer segmentation
    * data analysis for new dataset
    * dimensionality reduction
      * replace features with affinity of the instance to the cluster
    * anomaly detection
      * defect detection in manufacturing/fraud detection
    * semi-supervised learning
      * only a few labels are available, perform clustering, then propagate the labels to instances in the same cluster
    * search engine
    * segment image
      * cluster pixels according to their colors, then replace each pixel by their mean color

  * Algorithm

    * ​	K-means

      * idea & algorithm
* 
      * sklearn.cluster.KMeans(n_clusters = , init = , n_init = )`
* `n_init`: number of initial conditions
        * `Kmeans.labels_`:  copy of the clustering labels
      * `Kmeans.cluster_centers_`: centroids of the cluster
        * `Kmeans.inertia`: the mean squared distance between each instance and its closest centroid
        * `init`
          * `k-means++` : initializes centroids distant from each other, default in `sklearn`
        * `algorithm`
          * `auto`: “elkan” variation using the triangle inequality
        * how to choose `n_clusters`
          * 'elbow rule'
            * plot inertia v.s. n_clusters
          * 'silhouette score'
            * mean 'silhouette coefficient' over all instances
              * 'silhouette coefficient' = $b-a/\max(a,b)$
                * $a$: mean distance to the other instances in the same cluster (it is the mean intra-cluster distance) 
                * $b$: the mean nearest-cluster distance, that is the mean distance to the instances of the next closest cluster
                * 'silhouette coefficient' = +1 ==> wll inside the cluster
                * 'silhouette coefficient' = 0 ==> close to boundary
                * 'silhouette coefficient' = -1 ==> in worng cluster
          * `sklearn.metrics.silhouette_score(X = , labels = )`
      * `sklearn.cluster.MiniBatchKMeans(n_clusters = )`
        * Instead of using the full dataset at each iteration, the algorithm is capable of using mini-batches, moving the centroids just slightly at each iteration.
      * computational complexity
        * linear with regard to number of instances, number of clusters, and number of dimensions.
      * drawbacks
        * may find sub-optimal
        * to find the right number of clusters is not easy
        * does not behave very well when the clusters have varying sizes, different densities, or non-spherical shapes.
          * scaling is necessary
      * usage
        * segmentation of cluster
        * preprocessing
        * semi-supervised learning
          * use K-means to find representaive instance
          * label the representative instances
          * label propagation
    * DBSCAN
      * Algorithm
        * 