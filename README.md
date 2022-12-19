# Machine-Learning
Field of study that gives conputers the ability to learn without explicit programed

## Table of contents
1. [Supervised Learning](#supervisedlearning)
    1. [Linear Regression](#linearregression)
    2. [Logistic Regression](#logisticregression)
    3. [Decision Trees](#decisiontrees)
    4. [Neural Networks](#neuralnetworks)
2. [Unsupervised Learning](#unsupervisedlearning)
    1. [K Means Algorithm](#kmeans)
    2. [Anomaly Detection](#anomalydetection)
    3. [Collaborative Filtering](#collaborate)
    4. [Content Filtering](#content)
    5. [Principal Component Analysis](#pca)
3. [Techniques for Better Models](#technics)
    1. [Initialization of Deep Learning](#initializationdeeplearning)
    2. [Gradient Descent Optimization](#gradientoptimal)
    3. [Hyperparameters Tuning, Batch Norm, Softmax](#tuning_batchnorm_softmax)
4. [Strategy for Machinel Learning Projects](#strategmlyproject)
5. [Convolutional Neural Network](#cnn)
    1. [Foundations of CNN](#fundamentalscnn)
    2. [Deep Convolutional Models](#deepcnnmodel)
    3. [Object Detection](#objectdetect)
    4. [Face Recognition](#facerecogn)
    5. [Neural Style Transfer](#neuraltransfer)

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
    - $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]} (L2 reg: +\frac{\lambda}{2m}||W||_2^2)$, where shape of $Z^{[l]}$ is $(n^{[l]}, m)$
    - $A^{[l]} = g^{[l]}(Z^{[l]})$, where shape of $A^{[l]}$ is $(n^{[l]}, m)$
- Backward propagation for layer l:
    - $dZ^{[l]} = A^{[l]} - Y$, where shape of $dZ^{[l]}$ is $(n^{[l]}, m)$
    - $dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]T} (L2 reg: +\frac{\lambda}{m}W^{[l]})$, where shape of $dW^{[l]}$ is $(n^{[l]}, n^{[l-1]})$
    - $db^{[l]} = \frac{1}{m}np.sum(dZ^{[l]},\  axis=1,\  keepdims=True)$, where shape of $db^{[l]}$ is $(n^{[l]}, 1)$
    - $dA^{[l-1]} = W^{[l]T}dZ^{[l]} * g^{'[l-1]}(Z^{[l-1]})$, where shape of $dA^{[l]}$ is $(n^{[l]}, m)$
- Gradient Descent:
    - $W^{[l]} = W^{[l]} - \alpha dW^{[l]}$, where shape of $W^{[l]}$ is $(n^{[l]}, n^{[l-1]})$
    - $b^{[l]} = b^{[l]} - \alpha db^{[l]}$, where shape of $b^{[l]}$ is $(n^{[l]}, 1)$

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
    - Use better optimization algorithms (e.g. Adam, RMSprop)
    - Find better NN architectures/hyperparameters search (e.g. RNN, CNN)
- High Variance (Overfitting)
    - More data
    - Regularization 
        - L2 Regularization (if lambda increases, the algorithm would try to descrease the weights to keep the cost low (minimize cost) which would end up in a simpler regression or network.)
        - Dropout/Data augumentation
    - Find better NN architectures 
    
### Dropout Regularization (Inverted Dropout) <a name="dropout"></a>
- Randomly "dropping out" (i.e. setting to zero) a certain number of output units in a layer during training. 
    - d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob where a3 is the neurons in layer 3 and keep_prob is the probability a neuron is been kept.
    - a3 *= d3
    - a3 /= keep_prob (Since no dropout is implemented at test time, we want to keep the expected value of a3 so there's no scaling problem at test time)
- Intuition: any feature could be zero out, so the weight of the feature would be spread out (shrink)
- Make keep_prob lower at layers with more neurons to prevent overfitting

### Data Augumentation <a name="dataaug"></a>
- Technique used to artificially increase the size of a training dataset by creating modified versions of existing data (e.g. image: random rotation, distortion, color shifting)

### Normalizing Input <a name="normalinput"></a>
- Used to ensure that all of the input data is on the same scale, which can make the training process more efficient and improve the performance of the model.
- Use the same $\mu$ and $\sigma$ to normalize test set
- Make gradient descent less oscillate.

### Weight Initialization <a name="weightinit"></a>
- gradient is a measure of how much the output of a function changes when you change the inputs, and it's usually represented as a vector of partial derivatives
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

## Hyperparameters Tuning, Batch Norm, Softmax <a name="tuning_batchnorm_softmax"></a>

### Hyperparameter Tuning <a name="hyperparametertuning"></a>
- Use random search: select random values for the hyperparameters.
- Don't use grid search: some hyperparameters might not be as important so there's no point to try many values on them (e.g. tuning $\epsilon$).
- Coarse to fine: Starting with a broad range of values for the hyperparameters and then gradually narrowing down the range to identify the best values for the model. 
- Appropriate scale for hyperparameters, for example:
    - logrithmetic scale for $\alpha$, $\alpha \in [0.0001, 0.1]$, a = $log_{10}0.0001 = -4$, r = -4 * np.random.rand(), $\alpha = 10^r$ 
    - $\beta \in [0.9, 0.999]$, $1 - \beta \in [0.1, 0.001]$, r = -3 * np.random.rand(), $1 - \beta = 10^r$, $\beta = 1 - 10^r$ 

### Batch Normalization <a name="hyperparametertuning"></a>
- Used on the input layer and the hidden layers to improve the performance and stability of neural networks
    - Prevent covariate shift (input distribution changes): for activations in a hidden layers, the values of W and b in the previous layers keep on changing, so the activations suffer from covariate shift. Batch normalization ensures the mean and variance of z for the activations remains the same.
    - Slight regularization effect: since mean and variance is computed on each mini-batch instead of the entire dataset, it would add noise to the values of z in each mini-batch. 
- Given values in layer l, $z^{[l]\(1)}, ..., z^{[l]\(m)}$
    - $\mu = \frac{1}{m} \sum_{i} z^{[l]\(i)}$ 
    - $\sigma^2 = \frac{1}{m} \sum_{i} (z^{[l]\(i)} - \mu)^2$ 
    - $z_{norm}^{[l]\(i)} = \frac{z^{[l]\(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$
    - $\widetilde z^{[l]\(i)} = \gamma z_{norm}^{[l]\(i)} + \beta$ (you can also use gradient descent to update $\gamma\ and\ \beta$, different from the $\beta$ in gradient with momentum)
    - pass $\widetilde z^{[l]\(i)}$ to the activation function instead of $z^{[l]}$
- You can eliminate $b^{[l]}$ when calculating $z^{[l]}$, since $b^{[l]}$ will be canceled out by the mean subtraction step.
- Updates of $\gamma\ and\ \beta$ also works with gradient with momentum, RMSprop, Adam.
- At test time, use exponential weighted average across the mini-batches to calculate the mean and variance. and then calculate $z_{norm}$ and $\widetilde z$ for the test set.

### Softmax Regression <a name="softmax"></a>
- Used for multi-class classification
- Softmax function:
$$a_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} = P(y = j|\vec x)$$ where j is a specific number in K and K is the number of classes
- Cost function:
$$J = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^K y_j^{(i)}log\ \hat y_j^{(i)}$$ where m is the number of examples, K is the number of classes
- We want to maximize the likelihood that $y_j$ is one and $\hat y_j$ is close to one for all the training examples, i.e. maximize the likelihood that the target value and the predicted value are the same. 
- Then since gradient descent trys to minimize the cost, we add a negative sign at the front of the function. 
- For more numerical accuracy, use linear activation instead of softmax activation in the neural network, and add from_logits=True in the loss functoin when compile the model.

# Strategy for Machinel Learning Projects <a name="strategmlyproject"></a>
- Orthogonalization: making the different components of the model more independent from each other.

### Set Up Your Gaol <a name="strategmlyproject"></a>
- Use a single real number evaluate metric so you can quickly evaluate the performance of your model on the train/dev/test sets.
- For n metrics, it's reasonable to have 1 optimizing metric (e.g. maximize accuracy) and n-1 satisficing metric (e.g. thresholds such as runtime less than 100ms). 
- Make sure dev set and test set come from same distribution, and can be reflected to the data you expect to get in the future.
- If the dataset is large (e.g. 1 million examples), you can split the dataset to 98%/1%/1% as long as the test set is big enough to give high confidence in the overall performance of your system 
- You can add extra weights to the evaluate metric if some examples matters more (e.g. less violence image, add a weight indicates whether a picture contains violence)

### Human Level Performance <a name="humanlevel"></a>
- Huamn level performance is the ability of a model to perform a task at the same level of accuracy as a human.
- When a model's performance is less than human, there are a few tactics to improve the model 
    - get labeled data from human
    - gain insight from error analysis
    - analysis of bias/variance
- But after model performance surpass huamn, the model's progress usually slow down and becomes harder to apply the tactics. 
- Bias problem, when human level error is less than the training and dev error
- Bairance problem, when huamn level error is close to the training error, but varies from the dev error
- Use human level error as a proxy for the bayes error(owest possible prediction error that can be achieved)
    - in many tasks, human error is close to the bayes error
- Some tasks machine learning can pass human performance easily, e.g. product recommendations
    - structure data, not natural perception, lots of data

### Error Analysis <a name="erroranalysis"></a>
- Manually examine the mistakes that the algorithm is making to gain insights into what to do next.
- e.g. randomly choose 100 misclassified examples from the dev set, and examine which kind of examples has caused the most error
- DL algorithms are quite robust to random errors (incorrectly labeled examples) in the training set, but not systematics error (consistently mislabel  examples with a certain feature)
- build your first system quickly, then iterate

### Missmatched Training and Dev/Test Set <a name="mismatchedtrain"></a>
- If only a small amount of data is from the distribution A which you care about and a large amount of data is from another distributions B, then you should put all data from distribution B into training and split the data from distribution A into all training, dev, and test set.
    -  if 200000 exmaples from distribution B, 10000 exmaples from distribution A
    -  training set: all 200000 examples from B, and 5000 exmaples from A
    -  dev set: 2500 exmaples from A, test set: 2500 exmaples from A
- If the training error is much higher than the dev error, then it could be either a high variance problem or the training set is from different distribution to the dev set.
    - add a new training-dev set (same distribution as the training set, but not used for training)
    - If training-dev error is much higher than the training error, then variance problem
    - If training-dev error is close to the training error, but much lower than the dev error, then data mismatch(different distribution) problem
- Another way to address the data mismatch problem is to use the artificial data synthesis, e.g. creat more data similar to the dev set by addding car noise to the original audio. 

### Learning From Multiple Tasks <a name="multitask"></a>
- Transfer learning: A model trained on one task is used to perform a different but related task. 
    - useful when:
        - both tasks have the same input (e.g.both has image input)
        - not enough data available to train a model from scratch on the second task. 
        - low level feature learned from the first task is helpful to the second task 
    - if not enough data, you may freeze all the layers expect the output layer, and just train the parameters accosiate with the output layer
    - if more data is obtained, you may unfreeze a few more hidden layers previous from the output layers and train parameters on the unfreezed layers
    - if lots of data, you may use the parameters of the pre-train model as initializations and train the whole network using the data you have.
- Multi-task learning: - Used when you want to predict mulitple objects at the same time (unlike softmax, one image can have multiple label)
    - tasks which have shared lower-level feature
    - amound of data for each task is usually quite similar
    - as long as the neural network is big enough, it should perform similar or better than creating multiple neural networks for each task

### End to End Deep Learning <a name="endtoend"></a>
- Using deep neural networks to learn directly from raw inputs to outputs, without the need for manual feature engineering. 
- Can be more effective than traditional machine learning methods that rely on hand-crafted features because it allows the model to learn the most relevant and useful features directly from the data.
- Can lead to improved performance and faster, more efficient training.
- Need large amounts of data and the potential for overfitting.

# Convolutional Neural Network <a name="cnn"></a>

## Foundations of CNN <a name="fundamentalscnn"></a>
- CNN is particularly well-suited for image and video processing tasks
    - a convolutional layer would have a lot fewer parameters because
        - Parameter sharing: a feature detector that's useful in one part of the image is probably useful in another part of the image
        - Sparsity of connection: in each layer, each output value depends only on a small number of input
     - make the model more computationally efficient, and less prone to overfitting since it reduces the number of parameters that the model needs to learn
    - good at capturing translation invariance: an image shifted a few pixels should result in pretty similar features and should probably be assigned the same output label. 
        - convolutional layers with shift invariant filters that are able to recognize patterns and objects regardless of their position within the input data.
    - work well in computer vision
    - values in the filters are treated as parameters that deep neural networks can learn, and the model could be more rebust
    
### Padding: 
- Used in convolutional neural networks to add extra pixels to the edges of an image
- Solve the problem of shrinking output and throwing away info at the edges
- Valid convolution: no padding, input shape $(n_h, n_w)$, filter shape (f, f), output shape $(n_h - f + 1, n_w - f + 1)$
- Same convolutoin: pad so that the ouput size is the same as the input size.  output shape (n_h + 2p - f + 1, n_w + 2p - f + 1), p = $\frac{f - 1}{2}$, filter size is usually a odd number

### Stride:
- Number of pixels that the convolutional kernel (filter) moves when it slides across the input image.
- Input shape $(n_h, n_w)$, filter shape (f, f), output shape $(\lfloor \frac{n_h + 2p - f}{s} \rfloor + 1 ,\lfloor \frac{n_w + 2p - f}{s} \rfloor + 1)$

### Convolutions Over Volumn:
- Input shape $(n_h, n_w, n_c)$, filter shape $(f, f, n_c)$, output shape $(\lfloor \frac{n_h + 2p - f}{s} \rfloor + 1 ,\lfloor \frac{n_w + 2p - f}{s} \rfloor + 1, n_c^{'})$, where $n_c$ is the number of channel/depth and $n_c^{'}$ is the number of filters

### Convolution Layer l:
- Linear regression: wx+b, CNN: X * filter + b  
- Input shape $(n_h^{[l-1]}, n_w^{[l-1]}, n_c^{[l-1]})$
- Filter shape $(f^{[l]}, f^{[l]}, n_c^{[l-1]})$
- Output shape $(n_h^{[l]} = \lfloor \frac{n_h^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} \rfloor + 1 ,n_w^{[l]} = \lfloor \frac{n_w^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} \rfloor + 1, n_c^{[l]})$, where $n_c^{[l]})$ is the number of filters 
- Activations shape $(m, n_h^{[l]}, n_w^{[l]}, n_c^{[l]})$
- Weights shape $(f^{[l]}, f^{[l]}, n_c^{[l-1]}, n_c^{[l]})$
- Bias shape $(1, 1, 1, n_c^{[l]})$

### Pooling Layer
- Max pooling: reduces the dimensions$(n_h, n_w)$ of the input by taking the maximum value of a group of adjacent pixels in the input image.
- Intuition: if a feature is detected anywhere in the filter then keep a high number, if not, max number would still be quite small
- Input shape $(n_h, n_w, n_c)$, output shape $(\lfloor \frac{n_h + 2p - f}{s} \rfloor + 1 ,\lfloor \frac{n_w + 2p - f}{s} \rfloor + 1, n_c)$, p = 0 in most cases
- Average pooling: reduces the dimensions of the input by taking the average value of a group of adjacent pixels in the input image.
- Hyperparameters: filter size f, stride s, no parameters for gradient descent to learn

### Common Pattern
- Going into deeper layers of CNN, usually $n_h, n_w$ would decrease, and $n_c$ would increase
- Common pattern: one or more CONV -> POOL -> one or more CONV -> POOL -> FC -> FC -> FC -> softmax

## Deep Convolutional Models <a name="deepcnnmodel"></a>

### LeNet-5: 
- Used to recognize hand-written digits
- Image -> $\text{CONV}(n_c=6,f=5,s=1)$ -> $\text{AVG-POOL}(f=2,s=2)$ -> $\text{CONV}(n_c=16,f=6,s=1)$ -> $\text{AVG-POOL}(f=2,s=2)$ -> FC -> FC -> Softmax

### AlexNet
- Image(227, 227, 3) -> $\text{CONV}(n_c=96,f=11,s=4)$ -> $\text{MAX-POOL}(f=3,s=2)$ -> $\text{CONV}(\text{"same"},n_c=256,f=5,s=1)$ -> $\text{MAX-POOL}(f=3,s=2)$ -> $\text{CONV}(\text{"same"},n_c=384,f=3,s=1)$ -> $\text{CONV}(\text{"same"},n_c=384,f=3,s=1)$ -> $\text{CONV}(\text{"same"},n_c=384,f=3,s=1)$ -> $\text{MAX-POOL}(f=3, s=2)$ -> FC -> FC -> Softmax
- Similar to LeNet5, but much bigger and uses ReLU activations instead of sigmoid/tanh

### VGG-16
- Image(224, 224, 3) -> $\text{CONV}(\text{"same"},n_c=64,f=3,s=1)*2$ -> $\text{MAX-POOL}(f=2,s=2)$ -> $\text{CONV}(\text{"same"},n_c=128,f=3,s=1)*2$ -> $\text{MAX-POOL}(f=2,s=2)$ -> $\text{CONV}(\text{"same"},n_c=256,f=3,s=1)*3$ -> $\text{MAX-POOL}(f=2,s=2)$ -> $\text{CONV}(\text{"same"},n_c=512,f=3,s=1)*3$ -> $\text{MAX-POOL}(f=2,s=2)$ -> $\text{CONV}(\text{"same"},n_c=512,f=3,s=1)*3$ -> $\text{MAX-POOL}(f=2,s=2)$ -> FC -> FC -> Softmax

### Vanishing Gradients
- Very deep networks often have the problem of vanishing gradients
    - gradient is a measure of how much the output of a function changes when you change the inputs, and it's usually represented as a vector of partial derivatives
    - as we backprop from the output layer to the first layer, the gradients are multiplied by the weights on each layer and passed on to the previous layer
    - if the weights of the layers are small, the gradients can become very small (vanishing gradients) as they pass through the network, and the updates to the parameters will be very small as well. This can lead to slow convergence and poor performance.
    - small gradients may also make the updates to the parameters become very sensitive to small changes in the gradients. This can lead to oscillations or other unstable behavior during training
    - so with network depth increases, the accuracy could get saturated and the performance of the model could degrade

### ResNet
- Residual block allows you to train much deeper network with the training error keeps going down
- e.g.
    - main paht: $a^{[l]} \rightarrow \text{linear} \rightarrow \text{ReLU} \rightarrow \text{linear} \rightarrow \text{ReLU} \rightarrow a^{[l+2]}$
    - math expression: $a^{[l]} \rightarrow z^{[l+1]}=W^{[l+1]}a^{[l]}b^{[l+1]} \rightarrow a^{[l+1]}=g(z^{[l+1]}) \rightarrow z^{[l+2]}=W^{[l+2]}a^{[l+1]}b^{[l+2]} \rightarrow a^{[l+2]}=g(z^{[l+2]})$
    - then a shortcut/skip-connection from $a^{[l]}$ to $a^{[l+2]}$ that passes the value of $a^{[l]}$ to $a^{[l+2]}$ which makes $a^{[l+2]}=g(z^{[l+2]}+a^{[l]})$, and since $z^{[l+2]}+a^{[l]}$ need to have same dimension, "same" convolutional layers are used. 
    - if $a^{[l]}$ has smaller dimension, you can have $Wa^{[l]}$ to increase the dimension where W is $a^{[l]}$ with zero padding or use the convolutional block. 
- Suppose L2 regularization is used, and weight and bias shrinks to 0. Also, all values of A>0 because of ReLU activation (max(0, z))
- Then, $a^{[l+2]}=g(W^{[l+2]}a^{[l+1]}b^{[l+2]}+a^{[l]}) = g(a^{[l]}) = a^{[l]}$ which makes the identity function easy to learn for residual block 
- So adding a residual block in the middle or the end of a network doesn't hurt performance as the regularization will skip over them if those layers were not useful
- If the hidden layers in the residual block learns something useful, then the performance could be even better than just learning the identity function
![ResNet](https://user-images.githubusercontent.com/73056232/208222836-68096b1f-8d31-4bd2-a8ab-22bb95bc6a40.png)

### One by One Convolution
- Used to shrink the number of channels$(n_c)$ to increase computational efficiency 
- Input shape $(n_h^{[l-1]}, n_w^{[l-1]}, n_c^{[l-1]})$
- Filter shape $(1, 1, n_c^{[l-1]})$
- Output shape $(n_h^{[l-1]}, n_w^{[l-1]}, n_c^{[l]})$, where $n_c^{[l]}$ is the number of filters
- If $n_c^{[l]} == n_c^{[l-1]}$, then the effect is it just has nonlinearity and allows the model to learn a more complex function by adding another layer

### Inception Network
- Intuition: use a combination of 1x1, 3x3, and 5x5 convolutions and pooling to extract features from the input, and stacked the result on top of each other.
- Bottleneck layer: add a 1x1 convolution layer before the 3x3, or 5x5 convolutions can reduce the conputational cost significantly without hurting the performance
- Add a 1x1 convolution layer after the pooling layer to reduce the number of channels

### MobileNet
- Lower computional cost at deployment
- Useful for mobile and embedded vision applications
- Depthwise-separable convolution
    - reduces the number of parameters and the conputation cost while still preserving the ability to capture important features from the input.
    - depthwise convolution: process each channel of the input separately, using a separate convolutional kernel(filter) for each channel.
    - pointwise convolution: 1x1 convolution
    - pattern: input shape $(n_h, n_w, n_c)$ -> depthwise convolution with $n_c$ separated filters of shape (f, f)-> output $(n_{out}, n_{out}, n_c)$ -> pointwise convolution with filter shape $(1, 1, n_c)$ -> output shape $(n_{out}, n_{out}, n_c^{'})$
- A normal convolution would be: input shape $(n_h, n_w, n_c)$ -> filters shape $(f, f, n_c)$ -> output shape $(n_{out}, n_{out}, n_c^{'})$, and it will be a lot more conputational expensive than the depthwise-separable convolution

### EfficientNet
- Designed to achieve a high level of accuracy on image classification tasks while also being computationally efficient, using a combination of scaling techniques to adjust the depth, width, and resolution of the network.
- You can use the open source implementations of EfficientNet to find a good trade-off between depth, width, and resolution

## Object Detection <a name="objectdetect"></a>

### Object Localization
- Convolutional network with softmax output 
- Target value example: $y = [p_c, b_x, b_y, b_h, b_w, c_1, ..., c_i]$, where $p_c$ is whether an object is detected, $b_i$ are the parameters of the bounding box, and $c_i$ are the classes of the objects. 
- Top left corner of the image is (0, 0), and bottom right is (1, 1)
- If $p_c$ = 0, then the rest of value is negligible
- Loss function: $L(\hat y, y) = (\hat y_1 - y_1)^2 + (\hat y_2 - y_2)^2 + ... + (\hat y_8 - y_8)^2$ if $y_1 = 1, (\hat y_1 - y_1)^2$ if $y_1 = 0$
- In practice, common to use logistic loss for $p_c$, softmax loss for $c_i$, sqaure error for $b_i$ 
    
### Landmark Detection
- Target value example: $y = [p_c, l_{1x}, l_{1y}, ..., l_{ix}, l_{iy}]$, where $p_c$ is whether the object is detected, $l_{ix}, l_{iy}$ are the coordinates of the landmarks you want to recognize

### Object Detection
- Sliding window detection
    -  dividing the image or video into a grid of overlapping windows, and applying a detection algorithm (ConvNet) to each window to identify the presence of the desired object or feature, but it's computational expensive to do it sequentially.
    -  convolutional implementation:
        - trun FC layers into convolutional layers: treat the set of neurons in the first FC layer as a (1, 1, $n^{[l]}$) volume, and then use 1x1 filters to get (1, 1, $n^{[l+i]}$) volumes for the next FC layers 
        - e.g. suppose the ConvNet is trained with input shape (14, 14, 3) and output shape (1, 1, 4), and the test set has input shape (16, 16, 3). Then, if we just run the same ConvNet with the test set, the model will have a output of shape (2, 2, 4) which is consist of 4 (1, 1, 4) volumes that correspond to the results of 4 overlapping windows
        - with convolutional implementation, the algorithm makes all the predictions at the same time by one forward pass through the ConvNet

### Intersection Over Union (IoU)
- Used to evaluate the performance of an object detection algorithm. 
- It compares the area of overlap between the predicted bounding box and the ground-truth bounding box with the area of their union (the combined area of both boxes).
- $\frac{\text{size of the intersection}}{\text{size of the union}}$, "correct" if IoU $\geq$ threshold (0.5 commonly used)

### Non-Max Suppression
- Used to eliminate all the overlapping bounding boxes that have a high IoU to the bounding box with highest $p_c$ for each object (probability of an object is detected) so that the algorithm detects each object only once 
    - e.g. discard all bounding boxes with $p_c \geq 0.6$ 
    - while there are any remaining buonding boxes that haven't been visited:
        - pick the box with the highest $p_c$, and output that as a prediction 
        - discard any remaining box with IoU $\geq$ 0.5 with the box output in the previous step
- If there are k classes, then you should run non-max suppression k times independently for each of the classes

### Anchor Boxes
- Used for the cases where multiple objects are in the same grid cell
    - e.g. two anchor boxes
    - then, $y = [p_c, b_x, b_y, b_h, b_w, c_1, ..., c_i, p_c, b_x, b_y, b_h, b_w, c_1, ..., c_i]$ where the first half of y is the parameters for anchor box 1, and the second half is for anchor box 2
    - each object in training image is assigned to grid cell that contains object's midpoint and anchor box for the grid cell with highest IoU to the ground truth bounding box. It's assigned as pairs (grid cell number, anchor box number)

### YOLO (You Only Look Once) Algorithm
- Designed to perform object detection in real-time by processing images in a single pass. 
- Relatively effcient and accurate (works better than sliding window approach), and it can be used to detect a wide range of objects in an image.
    - divide the input image into a $n_h$ by $n_w$ grid, and for each grid cell, we have a training label y = mutiple $[p_c, b_x, b_y, b_h, b_w, c_1, ..., c_i]$ concatenate together depends on the number of anchor boxes
    - for a single pass through the ConvNet, we will get the output of shape $(n_h, n_w, n_p)$ where $n_h$ x $n_w$ is the number of grid cells and $n_p$ is the number of parameters in y
    - so, for each grid cell, we have a $(1, 1, n_p)$ volumn output that tells us if one or more objects are detected in that cell, and if so, wihch are the objects and what are the bounding box coordinates for the detected objects.
    - for an object showing up in multiple grid cells, assign it to the grid cell that contains the object's midpoint
    - Top left corner of each grid cell is (0, 0), and bottom right is (1, 1). $b_x, b_y$ have to be between 0 and 1, and $b_h, b_w$ could be greater than 1.
    - then get rid of low probability predictions, for each class, and run non-max suppression to generate the final predictions 

### Region Proposal: R-CNN
- Proposes a set of candidate regions in an image that may contain objects, and then classifies the regions to determine if they contain an object and, if so, what type of object it is.

### Semantic Segmentation with U-Net
- U-Net is commonly used fpr semantic segmentation tasks (assigning a semantic label (such as "car," "building," or "road") to each pixel in an image)
- As we going deeper into the U-net, $n_h, n_w$ increses and $n_c$ descrease
- Transpose convolution: used to increase $n_h, n_w$
    - increase the size of the output while preserving the spatial resolution of the input
    - opposite of normal convolution, it applies the filter on the output 
    - e.g. 

![tanspose-conv](https://miro.medium.com/max/1400/1*faRskFzI7GtvNCLNeCN8cg.png)
- U-Net intuition: 
    - the first half of the network will sort of compress the image where the height and width of the layers decrease and depth/channel increases, so a lot of spatial information (data that describes the location, shape, and/or size of objects) which can be used to identify the location and size of objects in an image is lost
    - the second half of the network uses the transpose convolution to reverse the layers size back up to the size of the original input image because we want a detailed, pixel-level map (i.e. high resolution) of the objects in an image.
    - also, there are skip connections from the earlier layers to the later layers
    - the later layers have the high level contextual information (relationships between the various objects in the scene, e.g. a stove is typically used for cooking) which can help the model classify and identify objects in the image more accurately, but missing detailed spatial information because the height and width are just lower
    - the earlier layers have the high resolution, low level feature information (e.g. simple shapes, edges)
    - so with the skip connections, the later layers will have both lower resolution, but high level contextual information, as well as the lower level, but more detailed spatial information (can tells you the shape of an object or where the object is located)
    - in other words, earlier layers give you the high resolution pixel-level map of the shape or location of each object, and then the info from the earlier layers is pass to the later layers through skip connections, so the later layers can use that high resolution map and the high level contextual information from the previous layer to check if each pixel is belong to one of the classes
    - the ouput of U-Net has shape $(h, w, n_{class})$, so there are h x w pixels and each one of the $(1, 1, n_{class})$ volumn tells you how likely is it for the pixel to be each of the classes in $n_{class}$
    
## Face Recognition <a name="facerecogn"></a>
### One Shot Learning
- Learning from one example or a small number of examples to recognize the person again
- Siamese network: used for one-shot learning tasks.
    - two identical subnetworks, which are trained to process inputs and generate outputs in a similar way.
    - then pass the outputs of the subnetworks to a logistic layer. Given input $x^{(i)}, x^{(j)}$,
    $$\hat y = \sigma (\sum_{k=1}^n w_k |f(x^{(i)})_k - f(x^{(j)})_k| + b)$$, and there are other ways to compute $|f(x^{(i)})_k - f(x^{(j)})_k|$
    - if $x^{(i)}$ is a real time image, and $x^{(j)}$ is the image from your database, then you can pre-compute all the images in the database using the subnetwork for $x^{(j)}$. The next time you want to compare two images, you only need to run the subnetwork for $x^{(i)}$
- Triplet loss: 
    - training data consists of triplets of examples, where each triplet consists of an anchor example, a positive example, and a negative example.
    - for each triplet, measure the distance between the anchor and positive example, and the distance between the anchor and negative example
    - since the anchor and positive image are from the same person, given f is the ouput of the network, we want $||f(anchor) - f(positive)||_2^2 - ||f(anchor) - f(negative)||_2^2 + \alpha \leq 0$ where $\alpha$ is a hyperparameter added to avoid the trivial solutions where f just set everything to 0 or set everything to be equal to each other 
    - cost function:
    $$J = \sum_{i=1}^m max(||f(anchor) - f(positive)||_2^2 - ||f(anchor) - f(negative)||_2^2 + \alpha, 0)$$
    - if anchor, positive, negative examples are randomly choosen during training, then $||f(anchor) - f(positive)||_2^2 - ||f(anchor) - f(negative)||_2^2 + \alpha \leq 0$ is easily satisfied, so the network won't learn much from it
    - choose the triplets where $d(anchor, positive) \approx d(anchor, negative)$

## Neural Style Transfer <a name="neuraltransfer"></a>
- Neural style transfer allows you to apply the style of one image to the content of another image
- Given content image C, style image S, and generated image G
- Initialize G randomly
- Use gradient descent to minimize J(G), G = G - $\frac{\partial}{\partial G} J(G)$
- The cost function for the neural style transfer is:
$$J(G) = \alpha J_{Content}(C, G) + \beta J_{Style}(S, G)$$
    - $\alpha, \beta$ are hyperparameters
    - measure content: choose a middle layer to measure how different is the content of C to the content of G
        - lower layers capture low-level features such as edges and shapes, and it may not accurately reflect the overall content of the image
        - in higher layers, the content cost may be overly sensitive to specific details and may not generalize well to different images.
    - use a pre-trained ConvNet (e.g. VGG)
    - if $a^{[l]\(C)}$ and $a^{[l]\(G)}$ are similar, then both images have similar content
    - content cost function:
    - $$J_{Content}(C, G) = ||a^{[l]\(C)} - a^{[l]\(G)}||_2^2$$
    - measure style: choose some earlier and later layers l to measure how correlated are the activations across different channels 
        - take both low level and high level features correlations into account when computing style
    - style/gram matrix: 
        - $$G_{kk^{'}}^{[l]\(S)} = \sum_{i=1}^{n_h} \sum_{j=1}^{n_w} a_{ijk}^{[l]\(S)} a_{ijk^{'}}^{[l]\(S)}$$
        - $$G_{kk^{'}}^{[l]\(G)} = \sum_{i=1}^{n_h} \sum_{j=1}^{n_w} a_{ijk}^{[l]\(G)} a_{ijk^{'}}^{[l]\(G)}$$
    - style cost function
    - $$J_{Style}^{[l]}(S, G) = \frac{1}{(n_h^{[l]}, n_w^{[l]}, n_c^{[l]})^2} \sum_{k=1}^{n_c^{[l]}} \sum_{k^{'}=1}^{n_c^{[l]}} (G_{kk^{'}}^{[l]\(S)} - G_{kk^{'}}^{[l]\(G)})^2$$
    - $$J_{Style}(S, G) = \sum_{l}^{n_l} \lambda ^{[l]} J_{Style}^{[l]}(S, G)$$
