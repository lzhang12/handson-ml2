# Chapter 6

## Decision Trees

* General Description of Decision Tree

  * for regression, classification, multioutput task
  * fundamental components of Random Forests

* Making Prediction

  * `sklearn.tree.DecisionTreeClassifier`

  * impurity function

    * Gini impurity

      $G_i =  1 - \sum\limits_{k=1}^n p_{i,k}^2$

      * $p_{i,k}$ is the ratio of class $k$ instances among the training instances in $i$th node

    * entropy

      $H_i = - \sum\limits_{i=1,p_{i,k}\neq0}^n p_{i,k}\log_2(p_{i,k})$

* Classification And Regression Tree(CART) Training Algorithm

  * cost function

    $J(k, t_k) = \frac{m_\text{left}}{m}G_\text{left} + \frac{m_\text{right}}{m}G_\text{right}$

    * feature $k$, threshhold $t_k$
    * note that the pair ($k, t_k$) is optimized so that the order of feature at each depth is determined
    * $m_\text{left/right}$ is the number of instances in the left/right subset

  * greedy algorithm, not guranteed to find the optimal solution

* Regularization Parameters

  * `max_depth`
  * `min_samples_split`: the minimum number of samples a node must have before it can be split
  * `min_samples_leaf`: the minimum number of samples a leaf node must have
  * `min_weight_fraction_leaf`: same as `min_samples_leaf`, but expressed as a fraction of the total number of weighted instances
  * `max_leaf_nodes` : maximum number of leaf nodes
  * `max_features`: maximum number of features that are evaluated for splitting at each node

* Regression

  * cost function

    $J(k, t_k) = \frac{m_\text{left}}{m}\text{MSE}_\text{left} + \frac{m_\text{right}}{m}\text{MSE}_\text{right}$

    * $\text{MSE}_\text{node} = \sum\limits_{i\in\text{node}} (\hat{y}_\text{node} - y^{(i)})^2$
    * $\hat{y}_\text{node} = \dfrac{1}{m_\text{node}} \sum\limits_{i\in\text{node}} \hat{y}^{(i)}$

* Limitations
  * Decision trees love orthogonal decision boundaries ==> sensitive to rotation
    * use PCA to result in a better orientation
  * Sensitive to small variation in the dataset