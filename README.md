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
- To train the model, squared error cost is used: 
$$Minimize_{w, b} \ J(w,b) = \sum_{i=1}^m(f_{w,b}(x^i) - y^i)^2$$

### Logistic Regression <a name="logisticregression"></a>
- Logistic regression is a classification method used to predict the probability of a binary outcome based on one or more independent variables.
- Logistic equation: h(x) = 1 / (1 + e^(-b0 - b1 * x1 - b2 * x2 - ... - bn * xn))
- We want to maximize the likelihood of the training data i.e. maximize the probability that the predicted value is the same as the target value, $(y * log(f(x)) + (1 - y) * log(1 - f(x)))$.
- To train the model, cross-entropy loss function is used:
$$Minimize_{w, b} \ J(w,b) = -1/m \sum_{i=1}^m (y^i * log(f_{w,b}(x^i)) + (1 - y^i) * log(1 - f_{w,b}(x^i)))$$

### Decision Trees <a name="decisiontrees"></a>
- Decision tree is used for both classification and regression
- Tree like structure where each branch of the tree represents a decision or rule that is used to predict the value of the dependent variable. 
- Two criterions to measure the impurity of a split and choose a feature to split:
    - given c is the number of classes, p is the proportion of the examples that belongs to class i for a particular node, and w is the proportion of examples in each branch.
    1. $$Entropy = \sum_{i=1}^c -p_ilog_2(p_i), \ \ \ \ \ Information\  Gain = H(p_1^{root}) - ((w^{left}H(p_1^{left}) + w^{right}H(p_1^{right}))$$
    2. $$Gini = 1 - \sum_{i=1}^c p_i^2$$ 

One downside of using only one decision tree is that small changes in the training set could result in a completely different decision tree, so creating multiple trees (tree ensembles) could make the algorithm more robust.
#### Random Forest Algorithm Intuition
- Using sampling with replacement to create a new training set of size m
- Train a decision tree on the new dataset, and when choosing a feature to split, pick a random subset of k < n features for the algorithm to choose from
- Repeat the process B times (common choice: 64, 128)
#### Boosted Trees Intuition
- When creating a new training set, make it more likely to pick misclassified examples from previously trained trees

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

