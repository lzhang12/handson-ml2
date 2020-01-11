# Chapter 4  Training Models

1. Linear Regression

   * prediction of linear regression

     $$\hat{y} = \boldsymbol{\theta}^T \boldsymbol{x}, \qquad (y = \text{prediction}, \boldsymbol{\theta} = \text{model parameters}, \boldsymbol{x} = \text{features})$$

   * Mean Squared Error (MSE) cost function for linear regression

     $$\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m} \sum\limits_{i=1}^m \big(\boldsymbol{\theta}^T \boldsymbol{x}^{(i)} - y^{(i)}\big)^2$$

     or $$\frac{1}{m} (\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y})^T (\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y})$$

     * $m$: number of instances

   * Normal equation (analytical solution) for linear regression

     $$ \hat{\boldsymbol{\theta}} = \big(\boldsymbol{X}^T \boldsymbol{X}\big)^{-1} \boldsymbol{X} \boldsymbol{y}$$

     * check [here][normal_eq_derive] for detailed derivation of the equation above, and further [here][matrix_calculus] for matrix calculus

     * In `scikit-learn`, the linear regression is based on the least square function `scipy.linalg.lstsq`  to compute the *pseudoinverse* of $\boldsymbol{X}$ using the *Singular Value Decomposition* technique (computational complexingty is $O(n^2)$, compared to $O(n^{2.4})$ to $O(n^3)$ for computing inverse).

2. Gradient Descent

   * GD step

     $$\boldsymbol{\theta}^{(\text{next step})} = \boldsymbol{\theta} - \eta \nabla_\boldsymbol{\theta} \text{MSE}(\boldsymbol{\theta})$$
   
   * MSE cost function is convex for linear regression, so GD is relatively fast
   
   * Batch GD
   
     * gradient vector
   
       $$ \nabla_\boldsymbol{\theta} \text{MSE} = \frac{2}{m}\boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{\theta} - \boldsymbol{y})$$
   
   * Stochastic GD
   
     * has better chance to find global minimum than Batch GD
     * learning schedule used to gradually reduce learning rate
   
   * Mini-batch GD
   
   * Comparison
   
     ![Comparison of algorithms for Linear Regression](https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/training_linear_models/Comparison_algorithms_Linear_Regression.png)
     
     * hyperparameters at least: learning rate and initial guess
   
3. Polynomial Regression

   * `from sklearn.preprocessing import PolynomialFeatures`
     * `PolynomialFeatures(degree=d)` transforms an array containing $n$ features into an array containing $\frac{(n+d)!}{n!d!}$ features.

4. Sources of Error

   * Bias
   * Variance
   * Irreducible error

5. Regularized Linear Model

   * Ridge Regression (Tikhonov regularization)

     * cost function

       $$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{2}(\Vert\boldsymbol{\omega}\Vert_2)^2, \qquad \boldsymbol{\omega} = (\theta_1, \theta_2, \cdots, \theta_n)^T$$

     * closed-form solution

       $$\hat{\boldsymbol{\theta}} = \big(\boldsymbol{X}^T \boldsymbol{X} + \alpha \boldsymbol{A}\big)^{-1} \boldsymbol{X} \boldsymbol{y}, \qquad \boldsymbol{A} \text{ is a } (n+1)\times(n+1) \text{ identity matrix with a 0 in the top-left corner}$$

   * LASSO Regression

     * Least Absolute Shrinkage and Selection Operator

     * cost function

       $$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha(\Vert\boldsymbol{\omega}\Vert_1), \qquad \boldsymbol{\omega} = (\theta_1, \theta_2, \cdots, \theta_n)^T$$

     * tends to completely eliminate the weights of the least important features

     * subgradient vector

       $$g(\boldsymbol{\theta}) = \nabla_\boldsymbol{\theta}\text{MSE} + \alpha \, \text{sign}(\boldsymbol{\omega}), \qquad \text{sign}(\theta_i) = \left\{ \begin{aligned} -1, \; \text{if} \; \theta_i < 0 \\ 0, \; \text{if}\; \theta_i = 0 \\ 1, \; \text{if} \; \theta_i > 0 \end{aligned} \right.$$

   * Elastic Net

     * cost function

       $$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r \alpha(\Vert\boldsymbol{\omega}\Vert_1) + (1-r)\frac{\alpha}{2}(\Vert\boldsymbol{\omega}\Vert_2)^2, \qquad  r = \text{mix ratio}$$

   * Early Stopping Regularization

     <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/training_linear_models/Early_Stopping_Regularization.png" alt="early stopping" style="zoom:35%;" />

     * `early_stopping` introduced in `SGDregressor` in sklearn 0.20

6. Logistic Regression

   * estimated probability

     $$\hat{p} = \sigma(\boldsymbol{x}^T\boldsymbol{\theta})$$, or $$\hat{\boldsymbol{p}} = \sigma(\boldsymbol{X} \boldsymbol{\theta})$$

     $$\sigma(t) = \frac{1}{1 + \exp(-t)}$$

   * model prediction

     $$\hat{y} = \left\{ \begin{aligned} 0, \; \hat{p} < 0.5 \\ 1, \; \hat{p} > 0.5 \end{aligned} \right.$$

   * cost function

     $$J(\boldsymbol{\theta}) = -\frac{1}{m} \big(\boldsymbol{y}^T \log(\hat{\boldsymbol{p}}) + (1 - \boldsymbol{y})^T \log(1 - \hat{\boldsymbol{p}})\big)$$

   * partial derivative

     $$\frac{\partial J}{\partial \boldsymbol{\theta}} = \frac{1}{m} \boldsymbol{X}^T (\hat{\boldsymbol{p}} - \boldsymbol{y})$$

   * regularization paramter in `sklearn` is $C = 1/\alpha$, where $\alpha$ is typically used in other regression methods to control the regularization strength

7. Softmax Regression/Multinomial Logistic Regression

   * generalized Logistic regression for multiple classes

   * Softmax score for class $k$

     $$s_k = \boldsymbol{x}^T\boldsymbol{\theta}^{(k)}$$

   * Softmax function

     $$\hat{p}_k = \sigma(\boldsymbol{s}(\boldsymbol{x}))_k = \frac{\exp(s_k(\boldsymbol{x}))}{\sum\limits_{j=1}^K \exp(s_j(\boldsymbol{x}))}$$

   * Softmax Regression classifier prediction

     $$\hat{y} = \text{argmax}\;\sigma(\boldsymbol{s}(\boldsymbol{x}))_k = \text{argmax}\;s_k(\boldsymbol{x}))$$

     * $\text{argmax}$ operator returns the value of a variable that maximizes function

   * Cross entropy cost function

     $$J(\boldsymbol{\Theta}) = -\frac{1}{m}\sum\limits_{i=1}^m \sum\limits_{k=1}^K \hat{y}_k^{(i)}\log(\hat{p}_k^{(i)})$$

     * $\boldsymbol{\Theta}$ is the parameter matrix, each row corresponds to $\boldsymbol{\theta}^{(k)}$
     * $\hat{y}_k^{(i)}$ is the target probability that $i$th instance belong to class $k$, either 0 or 1

   * Cross entropy gradient vector for $k$

     $$\nabla_{\boldsymbol{\theta}^{(k)}}J(\boldsymbol{\Theta}) =\frac{1}{m}\sum\limits_{i=1}^m (\hat{y}_k^{(i)}-\hat{p}_k^{(i)})\boldsymbol{x}^{(i)}$$

   

[normal_eq_derive]: https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression	"derivation of normal equation"
[matrix_calculus]: https://en.wikipedia.org/wiki/Matrix_calculus	"matrix calculus on wikipedia"