9# Machine-Learning
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
    1. [Initialization of Deep Learning](#initializationdeeplearning)
    2. [Gradient Descent Optimization](#gradientoptimal)
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
    - $\sigma(z) = \frac{1}{(1 + e^{-z})}$, tanh(z) = $\frac{(e^z - e^{-z})}{(e^z + e^{-z})}$, ReLU(z) = $max(0, z)$, leaky ReLU(z) = $max(0, z) + \alpha * min(0, z)$
    - If use a linear activation function, then no matter how many layers you have, all it's doing is computing a linear activation function.
- Forward propagation for layer l:
    - $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]} (L2 reg: +\frac{\lambda}{2m}||W||_2^2)$
    - $A^{[l]} = g^{[l]}(Z^{[l]})$
- Backward propagation for layer l:
    - $dZ^{[l]} = A^{[l]} - Y$
    - $dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]T} (L2 reg: +\frac{\lambda}{m}W^{[l]})$
    - $db^{[l]} = \frac{1}{m}np.sum(dZ^{[l]},\  axis=1,\  keepdims=True)$
    - $dA^{[l-1]} = W^{[l]T}dZ^{[l]} * g^{'[l-1]}(Z^{[l-1]})$
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

# Techniques to Improve Machine Learning Models <a name="technics"></a>

## Initialization of Deep Learning <a name="initializationdeeplearning"></a>
- High Bias (Underfitting)
    - Bigger networks
    - Train longer (e.g. run gradient descent longer)
    - Find better NN architectures 
- High Variance (Overfitting)
    - More data
    - Regularization (since we want to minimize the cost, if lambda increases, the algorithm would try to descrease the weights to keep the cost low which would end up in a simpler regression or network.)
    - Find better NN architectures 
    
### Dropout Regularization (Inverted Dropout) <a name="dropout"></a>
- Randomly "dropping out" (i.e. setting to zero) a certain number of output units in a layer during training. 
    - d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob where a3 is the neurons in layer 3 and keep_prob is the probability a neuron is been kept.
    - a3 *= d3
    - a3 /= keep_prob (Since no dropout is implemented at test time, we want to keep the expected value of a3 so there's no scaling problem at test time)
- Intuition: any feature could be zero out, so the weight of the feature would be spread out (shrink)
- Make keep_prob lower at layers with more neurons to prevent overfitting

### Data Augumentation <a name="dataaug"></a>
- Technique used to artificially increase the size of a training dataset by creating modified versions of existing data (e.g. image: random rotation, distortion)

### Normalizing Input <a name="normalinput"></a>
- Used to ensure that all of the input data is on the same scale, which can make the training process more efficient and improve the performance of the model.
- Use the same $\mu$ and $\sigma$ to normalize test set
- Make gradient descent less oscillate.

### Weight Initialization <a name="weightinit"></a>
- Proper weight initialization can help speed up training, improve the network's ability to learn, and prevent issues such as vanishing or exploding gradients
- In deep network (large number of layers), if $W^{[L]} > I$(Identity Matrix) the value of predicted y could be large (explode), $W^{[L]} < I$ the value of predicted y could be too small (vanish)
- For ReLU activation, use $W^{[l]} = np.random.rand(shape) + np.sqrt(\frac{2}{n^{[l-1]}})$
- For tanh activation, use $W^{[l]} = np.random.rand(shape) + np.sqrt(\frac{1}{n^{[l-1]}})$ (Xaviar initialization)
- OR $W^{[l]} = np.random.rand(shape) + np.sqrt(\frac{2}{n^{[l-1]} + n^{[l]}})$

### Gradient Checking <a name="gradientcheck"></a>
- Take $W^{[1]}, b^{[1]}, W^{[l]}, b^{[l]}$ and $dW^{[1]}, db^{[1]}, dW^{[l]}, db^{[l]}$, and reshape them into vectors $\theta$ and $d\theta$
- for i in range(1, l):
    -  $d\theta_{approx}^{[i]} = \frac{J(\theta_1, \theta_2, ..., \theta_i + \epsilon, ...) - J(\theta_1, \theta_2, ..., \theta_i - \epsilon, ...)}{2\epsilon}$
- Check if $d\theta_{approx} \approx d\theta$ by checking $\frac{\lVert d\theta_{approx} - d\theta \rVert_2}{\lVert d\theta_{approx}\rVert_2 + \lVert d\theta\rVert_2}$ $\approx$ 10^{-7}-Great, $\approx$ 10^{-5}-Alright, $\approx$ 10^{-3}-Worry
- Note: 
    - don't use in training, only to debug
    - if grad check fails, look at components to identity bug (e.g. value of grad check in some layers of $W^{[l]}$ or $b^{[l]}$ are very different)
    - remember regularization
    - don't work with dropout (you can check if grad check works without dropout first, and then turn on dropout if grad check pass)
    - run grad check at random initialization, then again after some iteration

## Gradient Descent Optimization  <a name="gradientoptimal"></a>
### Mini-Batch Gradient Descent <a name="minigradientdescent"></a>
- Used for faster training on large dataset. Divide the training examples into smaller subsets and run gradient descents on the subsets.
e.g. for t in range(1, 5000): (each t is a subset of the training example)
    - forward propagation on $X^{{t}}$
    - compute cost for $X^{[t]}, Y^{[t]}$ and divide cost by 1000 (if each subset contains 1000 training examples)
    - backward propagation using $X^{[t]}, Y^{[t]}$ to compute gradient
    - update W, b
- epoch: a single pass through the training set
    - in this case, 1 epoch allows you to take 5000 gradient descent steps
- batch size = m -> batch gradient descent (decrease on every iteration, but slow if m is large). if m <= 2000, use batch gradient descent
- batch size = 1 -> stochastic gradient descent (oscillate a lot, won't reach minimum)
- common mini-batch size: 64, 128, 256, 512 

### Exponentially Weighted Moving Average <a name="exponentialweightedavg"></a>
- Used to smooth out data and make it easier to identify trends
- Formula: $v_t = \beta v_{t-1} + (1 - \beta)\theta_t$, where $v_t$ is average over the last $\approx \frac{1}{1-\beta}$ data
- Bias correction: since $v_0 = 0$, $v_t$ would be small during initial phase. You can make $v_t = \frac{v_t}{1-\beta^t}$ to solve this problem

### Gradient with Momentum <a name="gradientmomentum"></a>
- Use exponentially weighted average to average out the last $\approx \frac{1}{1-\beta}$ gradients, so gradient descent is less oscillate
- On iteration t,
    - Compute dW, db on the current mini-batch
    - $v_{dW} = \beta v_{dW} + (1 - \beta)dW$ 
    - $v_{db} = \beta v_{db} + (1 - \beta)db$ 
    - W -= $\alpha v_{dW}$, b -= $\alpha v_{db}$ 
- Hyperparamters: $\alpha,\ \beta$ (common: $\beta = 0.9$)
- Bias correction is not necessary

### RMSprop <a name="rmsprop"></a>
- Intuition: Suppose b is the vertical axis, w is the horizontal axis, and gradient descent oscillate a lot vertically.
- On iteration t,
    - Compute dW, db on the current mini-batch
    - $S_{dW} = \beta S_{dW} + (1 - \beta)dW^2$ (element-wise square to make $S_{dW}$ relatively small to speed up the update in the horizontal direction)
    - $S_{db} = \beta S_{db} + (1 - \beta)db^2$ (element-wise square to make $S_{db}$ relatively large to slow down update in the vertical direction)
    - W -= $\alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon}$, b -= $\alpha \frac{db}{\sqrt{S_{db}} + \epsilon} (\epsilon$ avoid zero division, e.g. $\epsilon = 10^{-8}$)

### Adam Optimization <a name="adamoptimization"></a>
- Combine gradient with momentum and RMSprop
- $V_{dw} = 0,\ S_{dw} = 0,\ V_{db} = 0,\ S_{db} = 0$
- On iteration t,
    - Compute dW, db on the current mini-batch
    - $V_{dW} = \beta_1 V_{dW} + (1 - \beta_1)dW$, $V_{db} = \beta_1 V_{db} + (1 - \beta_1)db$  
    - $S_{dW} = \beta_2 S_{dW} + (1 - \beta_2)dW^2$, $S_{db} = \beta_2 S_{db} + (1 - \beta_2)db^2$
    - $V_{dW}^{corrected} = \frac{V_{dW}}{(1 - \beta_1^t)}$, $V_{db}^{corrected} = \frac{V_{db}}{(1 - \beta_1^t)}$
    - $S_{dW}^{corrected} = \frac{S_{dW}}{(1 - \beta_2^t)}$, $S_{db}^{corrected} = \frac{S_{db}}{(1 - \beta_2^t)}$
    - W -= $\alpha \frac{V_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}} + \epsilon}$, b -= $\alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}} + \epsilon}$
- Common hyperparamters: $\alpha(need\ tuning),\ \beta_1  = 0.9,\ \beta_2  = 0.999,\ \epsilon  = 10^{-8}$

### Learning Rate Decay <a name="learnratedecay"></a>
- Slowly reduce learning rate to speed up learning algorithm
- e.g. $\alpha = \frac{1}{1 + decayrate * epochnum}\alpha_0$, or $\alpha = 0.95^{epochnum}\alpha_0$, or $\alpha = \frac{k}{\sqrt{epochnum}}\alpha_0$, or $\alpha = \frac{k}{\sqrt{t}}\alpha_0$

## Hyperparameter Tuning <a name="hyperparametertuning"></a>
- Use random search: select random values for the hyperparameters.
- Don't use grid search: some hyperparameters might not be as important so there's no point to try many values on them (e.g. tuning $\epsilon$).
- Coarse to fine: Starting with a broad range of values for the hyperparameters and then gradually narrowing down the range to identify the best values for the model. 
- Appropriate scale for hyperparameters, for example:
    - logrithmetic scale for $\alpha$, $\alpha \in [0.0001, 0.1]$, a = $log_{10}0.0001 = -4$, r = -4 * np.random.rand(), $\alpha = 10^r$ 
    - $\beta \in [0.9, 0.999]$, $1 - \beta \in [0.1, 0.001]$, r = -3 * np.random.rand(), $1 - \beta = 10^r$, $\beta = 1 - 10^r$ 

## Batch Normalization <a name="hyperparametertuning"></a>
- Used on the input layer and the hidden layers to improve the performance and stability of neural networks
- Given values in layer l, $z^{[l]\(1)}, ..., z^{[l]\(m)}$
    - $\mu = \frac{1}{m} \sum_{i} z^{[l]\(i)}$ 
    - $\sigma^2 = \frac{1}{m} \sum_{i} (z^{[l]\(i)} - \mu)^2$ 
    - $z_{norm}^{[l]\(i)} = \frac{z^{[l]\(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$
    - $\widetilde z^{[l]\(i)} = \gamma z_{norm}^{[l]\(i)} + \beta$ (you can also use gradient descent to update $\gamma\ and\ \beta$, different from the $\beta$ in gradient with momentum)
    - pass $\widetilde z^{[l]\(i)}$ to the activation function instead of $z^{[l]}$
- You can eliminate $b^{[l]}$ when calculating $z^{[l]}$, since $b^{[l]}$ will be canceled out by the mean subtraction step.
- Updates of $\gamma\ and\ \beta$ also works with gradient with momentum, RMSprop, Adam.
