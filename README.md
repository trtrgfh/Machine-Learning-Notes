# Machine-Learning
Field of study that gives conputers the ability to learn without explicit programed

## Table of contents
- [Supervised Learning](#supervisedlearning)
    1. [Linear Regression](#linearregression)
    2. [Logistic Regression](#logisticregression)
    3. [Decision Trees](#decisiontrees)
    4. [Neural Networks](#neuralnetworks)
- [Unsupervised Learning](#unsupervisedlearning)
    1. [K Means Algorithm](#kmeans)
    2. [Anomaly Detection](#anomalydetection)
    3. [Collaborative Filtering](#collaborate)
    4. [Content Filtering](#content)
    5. [Principal Component Analysis](#pca)

- [Techniques for Better Models](#technics)

# Supervised Learning <a name="supervisedlearning"></a>
- Supervised learning is when a model is trained on a labeled dataset (dataset contains examples of inputs and their corresponding correct outputs), and the goal is to learn a mapping function from the input to the output. 
- Types of supervised learning: regression, classification

## Linear Regression <a name="linearregression"></a>
- Linear regression is used to predict the value of a continuous dependent variable based on one or more independent variables.
- Linear equation: y = b0 + b1 * x1 + b2 * x2 + ... + bn * xn
- To train the model, squared error cost is used: 
$$Minimize_{w, b} \ J(w,b) = \sum_{i=1}^m(f_{w,b}(x^i) - y^i)^2$$

## Logistic Regression <a name="logisticregression"></a>
- Logistic regression is a classification method used to predict the probability of a binary outcome based on one or more independent variables.
- Logistic equation: h(x) = 1 / (1 + e^(-b0 - b1 * x1 - b2 * x2 - ... - bn * xn))
- We want to maximize the likelihood of the training data i.e. maximize the probability that the predicted value is the same as the target value, $(y * log(f(x)) + (1 - y) * log(1 - f(x)))$.
- To train the model, cross-entropy loss function is used:
$$Minimize_{w, b} \ J(w,b) = -\frac{1}{m} \sum_{i=1}^m (y^i * log(f_{w,b}(x^i)) + (1 - y^i) * log(1 - f_{w,b}(x^i)))$$

## Decision Trees <a name="decisiontrees"></a>
- Decision tree is used for both classification and regression
- Tree like structure where each branch of the tree represents a decision or rule that is used to predict the value of the dependent variable. 
- Two criterions to measure the impurity of a split and choose a feature to split:
    - given c is the number of classes, p is the proportion of the examples that belongs to class i for a particular node, and w is the proportion of examples in each branch.
    1. $$Entropy = \sum_{i=1}^c -p_ilog_2(p_i), \ \ \ \ \ Information\  Gain = H(p_1^{root}) - ((w^{left}H(p_1^{left}) + w^{right}H(p_1^{right}))$$
    2. $$Gini = 1 - \sum_{i=1}^c p_i^2$$ 

One downside of using only one decision tree is that small changes in the training set could result in a completely different decision tree, so creating multiple trees (tree ensembles) could make the algorithm more robust.
### Random Forest Algorithm Intuition
- Using sampling with replacement to create a new training set of size m
- Train a decision tree on the new dataset, and when choosing a feature to split, pick a random subset of k < n features for the algorithm to choose from
- Repeat the process B times (common choice: 64, 128)
### Boosted Trees Intuition
- When creating a new training set, make it more likely to pick misclassified examples from previously trained trees

## Neural Networks <a name="neuralnetworks"></a>
- Neural networks are designed to mimic the way the human brain processes information.
- It is composed of multiple interconnected nodes, or "neurons," which process and transmit information. 
- Activation functions determine the output of a neuron given an input or set of inputs, some common activation functions are
    - $\sigma(z) = \frac{1}{(1 + e^{-z})}$, tanh(z) = $\frac{(e^z - e^{-z})}{(e^z + e^{-z})}$, relu(z) = $max(0, z)$, leaky relu(z) = $max(0, z) + \alpha * min(0, z)$
    - If use a linear activation function, then no matter how many layers you have, all it's doing is computing a linear activation function.
- Forward propagation for layer l:
    - $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$
    - $A^{[l]} = g^{[l]}(Z^{[l]})$
- Backward propagation for layer l:
    - $dZ^{[l]} = dA^{[l]} * g^{[l]'}(Z^{[l]})$
    - $dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]T}$
    - $db^{[l]} = \frac{1}{m}np.sum(dZ^{[l]},\  axis=1,\  keepdims=True)$
    - $dA^{[l-1]} = W^{[l]T}dZ^{[l]}$
- Gradient Descent:
    - $W^{[l]} = W^{[l]} - \alpha dW^{[l]}$
    - $b^{[l]} = b^{[l]} - \alpha db^{[l]}$

## Unsupervised Learning <a name="unsupervisedlearning"></a>
- Unsupervised learning is when a model is trained on an unlabeled dataset. The model is not provided with correct output examples for a given input, and the goal is to find hidden patterns or structures in the data. 
- Types of supervised learning: clustering, anomaly detection

### K Means Algorithm <a name="kmeans"></a>
- K means algorithm attempts to partition the dataset into k clusters and iteratively updating the cluster centroids and reassigning examples to their closest cluster, until the centroids converge. 
- Randomly pick K examples and set $\mu_1, ..., \mu_k$ equal to these examples.
- Cost function:
$$J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_k) = \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - \mu_c^{(i)}||^2$$
where $c^{(i)}$ is the index of cluster centroids closest to $x^{(i)}$ and $\mu_k$ is the average of points in cluster k.

### Anomaly Detection <a name="anomalydetection"></a>
- Anomaly detection algorithm is trained with dataset of normal events so that it learns to raise a red flag when encounters a unusual event
- The Gaussian (Normal) distribution is used in anomaly detection.
    - Bell shape curve
    - defined as: 
    $$p(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x-\mu)^2}{2\sigma^2}}\ \  Where\  \mu_j = \frac{1}{m}\sum_{i=1}^m x_j^{(i)}\  and\  \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m (x_j^{(i)} - \mu_j)^2$$
    - Anomaly if p(x) < $\epsilon$
### Collaborative Filtering <a name="collaborate"></a>
- Recommend items to you based on ratings of users who gave similar ratings as you
- Cost function:
$$Minimize_{(w^{(1)}, ..., w^{(n_u)}), (b^{(1)}, ..., b^{(n_u)}), (x^{(1)}, ..., x^{(n_m)})} =$$
$$\frac{1}{2} \sum_{(i,j):r(i,j)=1} (w^{(j)}x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2$$
Where given user j and item i, $n_u$ is the number of users, $n_m$ is the number of items, $n$ is the number of features, 
- Limitations:
    - cold start problem: recommend new items with few rates, and recommend items to new users who rated few items.
    - no natural way to use the side information about items or users

### Content Filtering <a name="content"></a>
- Recommend items to you based on features of user and item to find good match
- Cost function:
$$J = \sum_{(i,j):r(i,j)=1} (v_u^{(j)}v_m^{(i)} - y^{(i,j)})^2$$
Where (i,j) indicates whether user j has rated item i, $v_u$ is the output vecter of the user neural network and $v_m$ is the output vecter of the item neural network
- If the item dataset is large, two steps could help:
    - Retrieval: generate large list of plausible item candidates, i.e. a list tries to cover a lot of possible items you could recommend to the users
    - Ranking: use the retrival item list and feed the user vector and item vector to the neural networks to get the predicted ratings. Then, rank the items to display to the users.
- $v_m$ can be computed in advanced, so only need to compute $v_u$ when in use.

### Principal Component Analysis <a name="pca"></a>
- PCA is used to reduce the dimensionality of a dataset so the dataset is easier to visulize and understand. 
- PCA finds a new set of variables, called principal components, that are linear combinations of the original variables
- The new variables replace the original variables so that less varibles (dimensions) are used.

## Techniques to Improve Machine Learning Models <a name="technics"></a>


