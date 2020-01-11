# Chapter 5

1. Linear SVM Classification

   * "fitting the widest possible street between the classes" -- large margin classification
   * the decision boundary is fully determined by the instances located at the edge of the street (support vectors)
   * Hard margin vs Soft margin Classification
     * balance between keeping the street as large as possible, and limiting the margin violation
     * `sklearn.svm LinearSVC(C = , loss =)`
       * higher $C$ --> narrower decision boundary
       * using QP solver, faster than `sklearn.svm.SVC(kernel = 'linear')`
       * another way is  `sklearn.linear_model.SGDClassifier(loss = 'hinge', alpha = 1/(m*C))`
   
2. Nonlinear Classification

   * polynomial kernel
     * `sklearn.svm.SVC(kernel = 'poly', degree = , coef0 = )`
       * `coef0` controls how much the model is influenced by the high degree polynomials versus the low-degree polynomials
   * similarity function
     * `sklearn.svm.SVC(kernel = 'rbf', gamma = , C = )`
       * `gamma` is the std in the Gaussian RBF (radial basis function), smaller `gamma` narrower the Gaussian distribution, the more irregular the decision boundary.

3. Computational Complexity

   <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/svm/svm_comparison.png" alt="svm_complexity" style="zoom:50%;" />

4. SVM Regression

   * SVM Regression tries to fit as many instances as possible on the street while limiting margin violations (i.e., instances off the street)
   * `linear`
     * `sklearn.svm.LinearSVR(epsilon = )`
       * `epsilon` controls the width of the street, the larger `epsilon`, the wider the street
   * nonlinear
     * `sklearn.svm.SVR(kernel='poly', degree = , C = , epsilon = )`
       * `C` is the regularization parameter

5. Math Behind

   * $b$ is the bias term, $\boldsymbol{\omega}$ is the feature weight vector, and no bias feature added

   * linear SVM classifier prediction

     * $$\hat{y} = \left\{ \begin{aligned} 0 & \quad \text{if}  \; \boldsymbol{\omega}^T\boldsymbol{x} + b < 0,\\ 1 & \quad \text{if}  \; \boldsymbol{\omega}^T\boldsymbol{x} + b \geq 0 \end{aligned} \right. $$

   * Hard margin linear SVM classifier objective

     * $$\text{minimize} (\boldsymbol{\omega},b) \; \frac{1}{2}\boldsymbol{\omega}^T\boldsymbol{\omega}$$

       subject to $$t^{(i)} \, (\boldsymbol{\omega}^T\boldsymbol{x}^{(i)} + b) \geq 1$$ for $i = 1, 2, \cdots , m$

     * $t^{(i)} = \left\{\begin{aligned} -1, & \quad y^{(i)} = 0, \\ 1, & \quad y^{(i)} = 1\end{aligned}\right.$

   * Soft margin linear SVM classifier objective

     * $$\text{minimize} (\boldsymbol{\omega},b, \zeta) \; \frac{1}{2}\boldsymbol{\omega}^T\boldsymbol{\omega} + C\sum\limits_{i=1}^m \zeta^{(i)}$$

       subject to $$t^{(i)} \, (\boldsymbol{\omega}^T\boldsymbol{x}^{(i)} + b) \geq 1 -\zeta^{(i)}$$ and $\zeta^{(i)}\geq 0$ for $i = 1, 2, \cdots , m$

       * $\zeta^{(i)}$ is the slack variable, measuring how much the $i$th instance is allowed to violate the margin

   * Quadratic Programming Problem (QP problem)

     * minimize($\boldsymbol{p}$)  $\frac{1}{2} \boldsymbol{p}^T\boldsymbol{H}\boldsymbol{p}+ \boldsymbol{f}^T\boldsymbol{p}$

       subject to $\boldsymbol{A}\boldsymbol{p}\leq \boldsymbol{b}$

   * Dual Problem

     * minimize($\boldsymbol{\alpha}$)  

   * Kernalized SVM

   * Online SVM

     

   