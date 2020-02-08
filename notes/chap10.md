# Chapter 10 Introduction to Artificial Neural Networks with Keras

---
### Keywords
  * Perceptron
  * XOR problem
  * Sequential API
  * Functional API
---

* History
  * first introduced in 1943 by neurophysiologist Warren McCulloch and mathematician Walter Pitts

    * `A Logical Calculus of Ideas Immanent in Nervous Activity`

  * In the early 1980s, there was a revival of interest in connectionism (the study of neural networks), as new architectures were invented and better training techniques were developed.

  * why it is hot now

    <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/why_ann_hot_now.png" alt="why_ann_hot_now" style="zoom:30%;" />

* Artificial Neuron by McCulloch & Pitts

  * binary input/output

  * simple logic

    <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/ann_simple_logic.png" alt="ann_simple_logic" style="zoom:50%;" />

* Single Layer Perceptron 

  * invented by Frank Rosenblatt in 1957

  * Threshhold Logic Unit (TLU)

    <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/TLU.png" alt="TLU" style="zoom:50%;" />

  * bias neuron : always output 1

  * input neuron : passthrough

  * computation of  fully connected layer

    $$h_{\boldsymbol{W, b}}(\boldsymbol{X}) = \phi(\boldsymbol{X}\boldsymbol{W + \boldsymbol{b}})$$

    * $\boldsymbol{X}$ : n$\times$m matrix of input features. n = number of instances, m = number of features.
    * $\boldsymbol{W}$ : m$\times $p matrix of connection weights. p = number of artificial neurons. 
    * $\boldsymbol{b}$ : vector of connection weights between bias neuron and artificial neuron.
    * $\phi$ : activation function, if the neuron is TLU, it is step function

  * learning rule

    $$\omega_{i,j}^{\text{(next step)}} = \omega_{i,j} + \eta (y_j - \hat{y}_j) x_i$$

    * $\omega_{i,j}$ is the connection weight between $i$ input neuron and $j$ output neuron
    * $\eta$ is the learning rate
    * meaning : if the prediction agrees with the target, no need to update; otherwise update the weight according to the input value
    * equivalent to Stochastic Gradient Descent

  * limitations

    * <u>Exclusive OR problem</u>
    
  * `sklearn`

    * `sklearn.linear_model.Perceptron()`

* Multi-Layer Perceptron

  * structure

  <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/MLP.png" alt="MLP" style="zoom:50%;" />

  * solves the XOR problem

  * training algorithm -- backpropagation

    * for each training instance the backpropagation algorithm first makes a prediction (forward pass), measures the error, then goes through each layer in reverse to measure the error contribution from each connection (reverse pass), and finally slightly tweaks the connection weights to reduce the error (Gradient Descent step).
    * <u>automatic differentiation</u>

  * points

    * random initialization

    * activation function

      * logistic function $\sigma(z) = 1/(1+\exp(-z)) $ (0, 1)
      * hyperbolic tangent function $\tanh(z) = 2\sigma(2z)-1$ (-1, 1)
      * Rectified Linear Unit function $\text{ReLU}(z) = \max(0, z)$

      <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/activation_function.png" alt="activation_function" style="zoom:50%;" />

    * loss function
      * mean squared error
      * mean absolute error
      * Huber loss - combination of the two above

  * Regression problem

    * typical architecture of a regression MLP

      <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/regression_MLP_architecture.png" alt="regression_MLP_architecture" style="zoom:50%;" />

    * tensorflow.keras
    * california housing price example
      
      * `sklearn.datasets.fetch_california_housing`
  
* Classification problem
  
  * binary
  
    * single neuron with logistic activation, output gives a probability value
  
  * multilabel binary
  
    * multiple neurons for each label
  
  * exclusive multiclass
  
      * [softmax activation](./chap4.md)
  
    * typical architecture of a classification MLP
  
      <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/classification_MLP_architecture.png" alt="classification_MLP_architecture" style="zoom:50%;" />
      
    * `tensorflow.keras`
    
      * build a model
    
        `model = keras.models.Sequential([keras.layers.Flatten(input_shape = []), keras.layers.Dense(units = , activation = )])`
    
        * `activation ` = `"reLU"`, `"sigmoid"`, `"softmax"` et al.
    
      * compile
    
        `model.compile(loss = , optimizer = , metrics = )`
    
        * e.g., `loss = sparse_categorical_crossentropy` means using cross entropy as the loss function; `categorical` refers to the possibility of having more than two classes instead of binary; `sparse` refers to using a single integer from zero to the number of classes minus one (e.g. { 0; 1; or 2 } for a class label for a three-class problem), instead of a dense one-hot encoding of the class label (e.g. { 1,0,0; 0,1,0; or 0,0,1 } for a class label for the same three-class problem). For example, the actual class from a three-class problem is 1 (0,1,0), and our predicted probabilities for the three classes are [ 0.05, 0.80, 0.15 ]. The cross entropy for this particular prediction would be -log(0.8). ([from reddit](https://www.reddit.com/r/MLQuestions/comments/93ovkw/what_is_sparse_categorical_crossentropy/))
    
      * train & evaluate model
    
        `history = model.fit(X_train, y_train, epochs = , validation_data = (X_valid, y_valid))`
      
        * instead of passing a validation set, you can also set `validation_split`to the ratio of the training set
        * for classification problems, `class_weight` or `sample_weight` argument can be provided to the `fit` method. Explanation on [stackoverflow](https://stackoverflow.com/questions/48315094/using-sample-weight-in-keras-for-sequence-labelling)
      * `history.keys()` = `params, epoch, history`
    * pandas.DataFrame(history.history).plot(figsize = ())
      * predict
        * model.predict()
    
  * Non-sequential Neutral Network
  
    * example : wide and deep neutral network
  
      * deep and wide neural network
      <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/wide_DNN.png" alt="wide_DNN" style="zoom:50%;" />
      
      * multiple inputs
      <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/multiple_input_DNN.png" alt="wide_DNN" style="zoom:50%;" />
        
      * multiple outputs
        * example : locate (finding the coordinates of the object’s center, as well as its width and height) and classify (classification) the main object in a picture
      <img src="https://raw.githubusercontent.com/lzhang12/handson-ml2/master/images/ann/multiple_output_DNN.png" alt="wide_DNN" style="zoom:50%;" />

