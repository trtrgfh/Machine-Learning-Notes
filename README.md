# Machine-Learning
Field of study that gives conputers the ability to learn without explicit programed

## Table of contents
- [Supervised Learning](#supervisedlearning)
    1. [Linear Regression](#linearregression)
    2. [Logistic Regression](#logisticregression)
    3. [Decision Trees](#decisiontrees)
    4. [Neural Networks](#neuralnetworks)
- [Unsupervised Learning](#unsupervisedlearning)
    1. [K Nearest Neighbor](#knn)
    2. [Anomaly Detection](#anomalydetection)
    3. [Collaborative Filtering](#collaborate)
    4. [Content Filtering](#content)
- [Gradient Descent](#gradientdescent)

## Supervised Learning <a name="supervisedlearning"></a>
- Supervised learning is when a model is trained on a labeled dataset (dataset contains examples of inputs and their corresponding correct outputs), and the goal is to learn a mapping function from the input to the output. 
- Types of supervised learning: regression, classification

### Linear Regression <a name="linearregression"></a>
- Linear regression is used to predict the value of a continuous dependent variable based on one or more independent variables.
- Linear equation: y = b0 + b1 * x1 + b2 * x2 + ... + bn * xn
- The model is trained by fitting the linear equation to the training data using least squares optimization:
$$Minimize_{w, b} \ J(w,b) = \sum_{i=1}^m(f_{w,b}(x^{(i)}) - y^{(i)})^2$$

### Logistic Regression <a name="logisticregression"></a>
- Logistic regression is used to predict the probability of a binary outcome based on one or more independent variables.
- Logistic equation: h(x) = 1 / (1 + e^(-b0 - b1 * x1 - b2 * x2 - ... - bn * xn))
- The model is trained by adjusting the model's parameters to maximize the likelihood of the training data. 
- The cost function for logistic regression is given by the cross-entropy loss function:
$$cost(h(x), y) = -(y * log(h(x)) + (1 - y) * log(1 - h(x)))$$

Start with using the linear regression model, $f(x^{i}) = wx^{i} + b$, to predict y given x. Then, substitute the result from the linear regression function into sigmoid function which would map all inputs to values from 0 to 1.

### Decision Trees <a name="decisiontrees"></a>
The first paragraph text

### Neural Networks <a name="neuralnetworks"></a>
The first paragraph text

## Unsupervised Learning <a name="unsupervisedlearning"></a>
- Unsupervised learning is when a model is trained on an unlabeled dataset. The model is not provided with correct output examples for a given input, and the goal is to find hidden patterns or structures in the data. 
- Types of supervised learning: clustering, anomaly detection

### K Nearest Neighbor <a name="knn"></a>
The second paragraph text

### Anomaly Detection <a name="anomalydetection"></a>
The second paragraph text

### Collaborative Filtering <a name="collaborate"></a>
The second paragraph text

### Content Filtering <a name="content"></a>
The second paragraph text

## Gradient Descent <a name="gradientdescent"></a>
This is a sub paragraph, formatted in heading 3 style

