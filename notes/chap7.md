# Chapter 7

## Ensemble Learning

* Ensemble Learning

  * a group of predictors

* Voting Classifier

  * hard voting classifier
    * aggregate the predictions of each classifier and predict the class that gets the most votes
    * `sklearn.ensemble.VotingClassifier(estimator = , voting = 'hard')`
  * soft voting classifier
    * predict the class with the highest class probability, averaged over all the individual classifiers
    * `sklearn.ensemble.VotingClassifier(estimator = , voting = 'soft')`
      * all the estimators can estimate probability

* Bagging & Pasting

  * Bagging (**b**ootstrapping **ag**gregat**ing**)
    * use the same training algorithm for every predictor, but to train them on different random subsets of the training set
      * bootstrap[wiki] : any test or metric that relies on random sampling with replacement
    * `sklearn.ensemble.BaggingClassifier(base_estimator = , n_estimators = , max_samples = , bootstrap = True)`
  * Pasting
    * when sampling is performed without replacement
    * `sklearn.ensemble.BaggingClassifier(abse_estimator = , n_estimators = , max_samples = , bootstrap = False)`
  * benefit
    * predictor can be trained in parallel
    * prediction can be made in parallel
    * similar bias, but lower variance

* Out-of-Bag Evaluation

  * out-of-bag instances are instances not seen in the bagging process, out-of-bag evaluation is to evaluate the ensemble itself by averaging out the oob evaluations of each predictor
  * `sklearn.ensemble.BaggingClassifier(base_estimator, oob_score = True)`

* Random Patches & Random Subspaces

  * Random Patched
    * Sampling both training instances and features
    * `sklearn.ensemble.BaggingClassifier(bootstrap = True, bootstrap_features = True)`
  * Random Subspaces
    * Keeping all training instances, but sampling features
    * `sklearn.ensemble.BaggingClassifier(bootstrap = False, bootstrap_features = True)`
  * useful for high-dimensional inputs (e.g., images)
  * trading bias for variance

* Random Forests

  * an ensemble of Decision Trees, generally trained via the bagging method (or sometimes pasting), typically with max_samples set to the size of the training set
  * `sklearn.ensemble.RandomForestClassifier(n_estimators = , max_leaf_nodes =)`
    * equivalent to `sklearn.ensemble.BaggingClassifier(base_estimator = DecisionTreeClassifier(splitter = 'random', max_leaf_nodes=16), n_estimators=, max_samples = 1.0, bootstrap = True)`
  * Extra-Trees (Extremely Randomized Trees)
    * use random thresholds for each feature rather than search for the best
    * `ExtraTreesClassifier`
  * feature importance
    * `RandomForestClassifier.feature_importances_`

* Boosting

  * AdaBoost (Adaptive Boosting)

    * a first base classifier (such as a Decision Tree) is trained and used to make predictions on the training set. The relative weight of misclassified training instances is then increased. A second classifier is trained using the updated weights and again it makes predictions on the training set, weights are updated, and so on
    * `sklearn.ensemble.AdaBoostClassifier(base_estimator = , n_estimators = , algorithm = , learning_rate = )`
      * `algorithm = 'SAMME' or 'SAMME.R'`, `SAMME` (Stagewise Additive Modeling using Multiclass Exponential loss function)
    * give each instance a weight $w^{(i)}$, the new weight of the next predictor is updated by
      * $w^{(i)}_{j+1} = \left\{ \begin{aligned} & w_j^{(i)}, & \text{if} \; & \hat{y}^{(i)} = y^{(i)} \\ & w_j^{(i)} \exp(\alpha_j), & \text{if} \; & \hat{y}^{(i)} \neq y^{(i)} \end{aligned} \right.$
        * then it needs to be normalized
      * $\alpha_j = \eta\log\frac{1 - r_j}{r_j}$
        * $\eta$ is the learning rate
        * $r_j = \frac{\sum\limits_{i=1,\, \hat{y}_j^{(i)} = y^{(i)}}^m w_j^{i}}{\sum\limits_{i=1}^m w_j^{i}}$ is the weighted error rate
    * prediction
      * $\hat{y}(\boldsymbol{x}) = \text{argmax}\sum\limits_{j=1, \, \hat{y}_j = k}^N \alpha_j$

  * Gradient Boosting

    * similar to AdaBoost, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor
    * `sklearn.ensemble.GradientBoostingRegressor(max_depth = , n_estimators = , learning_rate = )`
      * early stopping: `GradientBoostingRegressor.staged_predict`
    * `XGBoost` package

  * Stacking

    * use different predictors to train part of the data, then using the predictions of the other part of the data as the training set for a new layer of predictor. More layers can be built.

      <img src="https://github.com/lzhang12/handson-ml/blob/master/images/ensembles/stacking.png?raw=true" alt="stacking" style="zoom:50%;" /> 