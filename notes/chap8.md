# Chapter 8

## Dimensionality Reduction

* Motivation

  * 1. To speed up training
    2. To visualize data
    3. To save space
  * Curse of Dimensionality

    * randomly sampled high-dimensional vectos are generally very sparse

    * Example

      *if you pick a random point in a unit square (a 1 × 1 square), it will have only about a 0.4% chance of being located less than 0.001 from a border (in other words, it is very unlikely that a random point will be “extreme” along any dimension). But in a 10,000-dimensional unit hypercube (a 1 × 1 × ⋯ × 1 cube, with ten thousand 1s), this probability is greater than 99.999999%. Most points in a high-dimensional hypercube are very close to the border.*

* Approach of Dimensionality Reduction

  * Projection

  * Manifold learning

    * manifol assumptio/manifold hypothesis

      *most real-world high-dimensional datasets lie close to a much lower-dimensional manifold.*

* Algorithms

  * PCA

    * idea

      *identify the axis that minimizes the mean squared distance between the original dataset and its projection onto that axis*

    * principal components

      * centering

      * Singular Value Decomposition

        $X = U \Sigma V^T$

    * projection of training set to $d$ dimensions

      $X_{d_\text{proj}} = X W_d$

      * $W_d$ is the first $d$ columns of $V$, $m\times d$ matrix

    * `sklearn.decomposition.PCA(n_components = , svd_solver = auto/full/randomized)`
      * if `n_components >= 1`, it means number of components you want to preserve; if `n_components < 1`, it means the ratio of variance you want to preserve
      * `PCA.components_`
      * `PCA.explained_variance_ratio_`
      * by default, `svd_solver` is set to `auto`: automatically uses randomized PCA when $m, n > 500$ and $d < 80\% \min(m,n)$.
    * Incremental PCA
      * online PCA
      * `sklearn.decomposition.IncrementalPCA.partial_fit`
        * split the data
        * use `memmap` class in `NumPy` to map a large array in file to memory
    * Kernel PCA
      * `sklearn.decomposition.KernelPCA(n_components = , kernel = , gamma = , fit_inverse_transform = )`
      * if `fit_inverse_transform` set to `True`, a `inverse_transform` method can be used to calculate the pre-image. Then pre-image can be compared with the original data to compute the reconstruction error and help evaluate the hyper parameters like `gamma`.
  
* Locally Linear Embedding (LLE)
  
  * `sklearn.manifold.LocallyLinearEmbedding(n_components = , n_neighbors = )`
  
  * the idea is to first measuring how each training instance linearly relates to its closest neighbors (c.n.), and then looking for a low-dimensional representation of the training set where these local relationships are best preser
  
  * mathematical details
  
    * step 1: linearly modeling local relationships
  
      $$\hat{\boldsymbol{W}} = \underset{\boldsymbol{W}}{\text{argmin}} \sum\limits_{i=1}^{m}\Big( \boldsymbol{x}^{(i)} - \sum\limits_{j=1}^m \omega_{i,j} \boldsymbol{x}^{(j)} \Big)^2$$
  
      subject to $$\left\{\begin{aligned} & \omega_{i,j} = 0, & &\text{if } \boldsymbol{x}^{(j)} \text{ is not the } k \text{ closest neighbors of } \boldsymbol{x}^{(i)}\\ & \sum\limits_{j=1}^{m}\omega_{i,j} = 1, & & \text{for } i = 1, 2, \cdots, m \end{aligned} \right.$$
  
      * $\text{argmin}$ tries to find the variable $\boldsymbol{W}$ or, $\omega_{i,j}$
  
    * step 2: reducing dimensionality while preserving relationships
  
      $$\hat{\boldsymbol{Z}} = \underset{\boldsymbol{Z}}{\text{argmin}} \sum\limits_{i=1}^{m}\Big( \boldsymbol{z}^{(i)} - \sum\limits_{j=1}^m \hat{\omega}_{i,j} \boldsymbol{z}^{(j)} \Big)^2$$
  
        * $\boldsymbol{z}^{(i)}$ is the image of $\boldsymbol{x}^{(i)}$ in the reduced space