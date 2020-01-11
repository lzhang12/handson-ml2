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

        <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/unspervised_learning/K-means_Algorithm.png?raw=true" alt="K-means_Algorithm" style="zoom:50%;" />

      * `sklearn.cluster.KMeans(n_clusters = , init = , n_init = )`
      
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
    
        * simple yet powerful algorithm, capable of identifying any number of clusters, of any shape, it is robust to outliers, and it has just two hyperparameters (eps and min_samples)
        * if the density varies significantly across the clusters, it can be impossible for it to capture all the clusters properly.
    
      *  computational complexity
    
        * $O(m\log m)$
    
      * `sklearn.cluster.DBSCAN`
    
        * no predict method, only fit_predict method
    
        * to predict with knn
    
          ```python
          knn = KNeighborsClassifier(n_neighbors=50)
          knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
          ```
    
    * Gaussian Mixture
    
      * A Gaussian mixture model (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown.
    
      * `sklearn.mixture.GaussianMixture(n_components = , n_init = , covariance_type = 'full/spherical/diag/tied')`
    
        * `weights_`
        * `means_`
        * `covariances_`
        * `converged_`
        * `n_iter_`
    
      * <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/unspervised_learning/Trained_GaussianMixture.png" alt="K-means_Algorithm" style="zoom:50%;" />
    
      * Anomaly detection
    
        ```python
        densities = gm.score_samples(X)
        density_threshold = np.percentile(densities, 4)
        anomalies = X[densities < density_threshold]
        ```
        
      * Selecting the number of clusters
      
        * Bayesian information criterion (BIC) and Akaike information criterion (AIC)
      
          $$BIC = \log(m) p - \log(\hat{L})$$
      
          $$AIC = 2p -\log(\hat{L})$$
      
          * $m$ = number of instances
          * $p$ = number of parameters learned by the model
          * $\hat{L}$ = maximized value of the likelihood function of the model 