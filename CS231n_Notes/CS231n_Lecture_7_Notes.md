# CS231n Lecture 7: Training Neural Networks, Part 2
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Administrative](#administrative)
- [Review of Previous Lecture](#review-of-previous-lecture)
- [Today's Topics](#todays-topics)
- [More Normalization](#more-normalization)
  - [Batch Normalization Review](#batch-normalization-review)
  - [Batch Normalization for ConvNets](#batch-normalization-for-convnets)
  - [Layer Normalization](#layer-normalization)
  - [Instance Normalization](#instance-normalization)
  - [Comparison of Normalization Layers](#comparison-of-normalization-layers)
  - [Group Normalization](#group-normalization)
  - [Decorrelated Batch Normalization](#decorrelated-batch-normalization)
- [Fancier Optimization](#fancier-optimization)
  - [Problems with SGD](#problems-with-sgd)
  - [SGD + Momentum](#sgd--momentum)
  - [Nesterov Momentum](#nesterov-momentum)
  - [AdaGrad](#adagrad)
  - [RMSProp](#rmsprop)
  - [Adam](#adam)
  - [Learning Rate Decay](#learning-rate-decay)
  - [First-Order vs Second-Order Optimization](#first-order-vs-second-order-optimization)
- [Regularization](#regularization)
  - [Early Stopping and Model Ensembles](#early-stopping-and-model-ensembles)
  - [L2, L1, and Elastic Net](#l2-l1-and-elastic-net)
  - [Dropout](#dropout)
  - [Data Augmentation](#data-augmentation)
  - [Other Regularization Patterns](#other-regularization-patterns)
- [Transfer Learning](#transfer-learning)
- [Summary & Key Takeaways](#summary--key-takeaways)

---

## Administrative
- Assignment 1 is being graded, stay tuned.
- Project proposals due tomorrow by 11:59pm on Gradescope.
- Assignment 2 is out, due Wednesday 5/2 11:59pm.

## Review of Previous Lecture
### Activation Functions
[KEY CONCEPT] **Activation Functions**: Non-linearities applied to the outputs of linear layers.
Formulas and shapes for various functions:
- **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **tanh**: $\tanh(x)$
- **ReLU**: $\max(0, x)$  
  > **Key Insight**: ReLU is a good default choice.
- **Leaky ReLU**: $\max(0.1x, x)$
- **Maxout**: $\max(w_1^T x + b_1, w_2^T x + b_2)$
- **ELU**: 
  $$ f(x) = \begin{cases} x & \text{if } x \ge 0 \\ \alpha(e^x - 1) & \text{if } x < 0 \end{cases} $$

### Weight Initialization
- **Initialization too small**: Activations go to zero, gradients also zero, no learning.
- **Initialization too big**: Activations saturate (e.g., for tanh), gradients become zero, no learning.
- **Initialization just right**: Nice distribution of activations at all layers, learning proceeds nicely.

### Data Preprocessing
- **Original data** vs **zero-centered data** vs **normalized data**.
- [DIAGRAM] Scatter plots showing data centering around origin and then scaled to have unit variance.
- Before normalization: classification loss very sensitive to changes in weight matrix; hard to optimize.
- After normalization: less sensitive to small changes in weights; easier to optimize.

### Babysitting Learning & Hyperparameter Search
- **Loss curves**: Very high learning rate causes explosion, high causes early plateau, low causes slow learning.
- **Hyperparameter search**: Grid Layout vs Random Layout. Random layout is better for finding optimal parameters when some parameters are more important than others. Coarse to fine search is recommended.

## Today's Topics
- More normalization
- Fancier optimization
- Regularization
- Transfer Learning

## More Normalization

### Batch Normalization Review
[KEY CONCEPT] **Batch Normalization**: Normalizes activations in a network across the minibatch.

**Training time:**
[FORMULA] 
Mean: $\mu_j = \frac{1}{N}\sum_{i=1}^N x_{i,j}$
Variance: $\sigma_j^2 = \frac{1}{N}\sum_{i=1}^N (x_{i,j} - \mu_j)^2$
Normalize: $\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \varepsilon}}$
Scale and shift: $y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j$
Parameters $\gamma, \beta$ are learnable.

**Test time:**
We cannot estimate mean and variance from a minibatch at test time.
Instead, use a running average of values seen during training.

### Batch Normalization for ConvNets
- For fully-connected networks, input is $N \times D$, $\mu$ and $\sigma$ are $1 \times D$.
- For convolutional networks (Spatial Batchnorm / BatchNorm2D), input is $N \times C \times H \times W$.
- $\mu, \sigma, \gamma, \beta$ are $1 \times C \times 1 \times 1$. Normalization happens across $N, H, W$ for each channel.

### Layer Normalization
[KEY CONCEPT] **Layer Normalization**: Normalizes across the feature dimension for each item in the batch.
- For fully-connected networks: Input $N \times D$, $\mu$ and $\sigma$ are $N \times 1$.
- Same behavior at train and test time! Can be used in recurrent networks.

### Instance Normalization
[KEY CONCEPT] **Instance Normalization**: Normalizes across the spatial dimensions for each channel and each item.
- Input is $N \times C \times H \times W$. $\mu$ and $\sigma$ are $N \times C \times 1 \times 1$.
- Same behavior at train / test! Often used in style transfer.

### Comparison of Normalization Layers
[DIAGRAM] 3D cubes representing feature maps ($N$ batch size, $C$ channels, $H,W$ spatial dimensions). Blue regions show the elements grouped for computing mean and variance:
- **Batch Norm**: normalizes across $N, H, W$.
- **Layer Norm**: normalizes across $C, H, W$.
- **Instance Norm**: normalizes across $H, W$.
- **Group Norm**: normalizes across groups of channels and $H, W$.

### Decorrelated Batch Normalization
- BatchNorm normalizes the data but cannot correct for correlations among input features.
- Decorrelated Batch Normalization (DBN) whitens the data using the full covariance matrix of the minibatch: $\hat{x}_i = \Sigma^{-\frac{1}{2}}(x_i - \mu)$.

## Fancier Optimization

### Problems with SGD
1. **High Condition Number**: Loss changes quickly in one direction and slowly in another.
   - [DIAGRAM] Oval contour lines of a loss function.
   - Very slow progress along shallow dimension, jitter along steep direction.
2. **Local Minima and Saddle Points**:
   - [DIAGRAM] 1D loss curves showing a local minimum (valley) and a saddle point (flat plateau).
   - Gradient is zero at these points, causing gradient descent to get stuck. Saddle points are much more common in high-dimensional spaces.
3. **Gradient Noise**: Gradients come from minibatches, so they can be noisy.

### SGD + Momentum
Builds up "velocity" as a running mean of gradients.
[FORMULA] 
$v_{t+1} = \rho v_t + \nabla f(x_t)$
$x_{t+1} = x_t - \alpha v_{t+1}$

[CODE]
```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x -= learning_rate * vx
```
> **Key Insight**: Momentum helps overcome local minima, saddle points, and poor conditioning by building up velocity in consistent directions. $\rho$ is typically 0.9 or 0.99.

### Nesterov Momentum
**Intuitive Explanation:**
Think of standard Momentum as a ball rolling blindly down a hill. It looks at the slope (gradient) at its *current* position and updates its velocity. The problem is that if it builds up a lot of speed, it will massively overshoot the bottom of the valley before it realizes the slope has changed direction.

Nesterov Momentum gives the ball a pair of binoculars. It says: "You already know your current velocity is going to carry you forward. Why don't you look ahead to where you are *going* to land, and calculate the gradient *there*?"
1. "Look ahead" by taking a hypothetical step using only the current velocity: $x_{ahead} = x_t + \rho v_t$.
2. Calculate the gradient at that future position: $\nabla f(x_{ahead})$.
3. Update the actual velocity using that "future" gradient.

This prevents massive overshooting because if the future position is starting to go uphill, the future gradient will tell the optimizer to hit the brakes *before* it actually makes the mistake.

[FORMULA]
$v_{t+1} = \rho v_t - \alpha \nabla f(x_t + \rho v_t)$
$x_{t+1} = x_t + v_{t+1}$

[DIAGRAM] Vector addition diagram showing Nesterov step. Standard momentum takes a step in velocity direction, then gradient direction. Nesterov takes a step in velocity direction, computes gradient at that future point, and adds it.

[CODE] Change of variables to make it fit standard API:
```python
dx = compute_gradient(x)
old_v = v
v = rho * v - learning_rate * dx
x += -rho * old_v + (1 + rho) * v
```

### AdaGrad
Added element-wise scaling of the gradient based on the historical sum of squares in each dimension.
[FORMULA]
$cache_{t+1} = cache_t + (\nabla f(x_t))^2$
$x_{t+1} = x_t - \frac{\alpha}{\sqrt{cache_{t+1}} + \epsilon} \nabla f(x_t)$

[CODE]
```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```
- Progress along "steep" directions is damped; progress along "flat" directions is accelerated.
- **Problem**: Over a long time, the step size decays to zero because the sum of squares only grows.

### RMSProp
Fixes AdaGrad's diminishing learning rate by using a moving average of squared gradients.
[FORMULA]
$cache_{t+1} = \rho \times cache_t + (1 - \rho) (\nabla f(x_t))^2$
$x_{t+1} = x_t - \frac{\alpha}{\sqrt{cache_{t+1}} + \epsilon} \nabla f(x_t)$

[CODE]
```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

### Adam
Combines Momentum and RMSProp.
[FORMULA]
$m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla f(x_t)$
$v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla f(x_t))^2$
$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^t}$
$\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^t}$
$x_{t+1} = x_t - \frac{\alpha}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$

[CODE]
```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    first_unbias = first_moment / (1 - beta1 ** t)
    second_unbias = second_moment / (1 - beta2 ** t)
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)
```
> **Key Insight**: Bias correction is necessary because first and second moments are initialized to zero. `beta1=0.9`, `beta2=0.999`, and `learning_rate=1e-3` or `5e-4` is a great starting point for many models.

### Learning Rate Decay
- SGD, SGD+Momentum, Adagrad, RMSProp, Adam all have learning rate as a hyperparameter.
- **Step decay**: decay learning rate by half every few epochs.
- **Exponential decay**: $\alpha = \alpha_0 e^{-kt}$
- **1/t decay**: $\alpha = \alpha_0 / (1 + kt)$
- Learning rate decay is more critical with SGD+Momentum and less common with Adam.

### First-Order vs Second-Order Optimization
- **First-Order**: Uses gradient to form linear approximation, steps to minimize approximation.
- **Second-Order**: Uses gradient and Hessian to form quadratic approximation, steps to minima.
  - [FORMULA] Newton parameter update: $\theta^* = \theta_0 - H^{-1} \nabla_\theta J(\theta_0)$
  - Nice because there are no hyperparameters or learning rates!
  - Bad for deep learning because computing and inverting the Hessian is computationally intractable ($O(N^3)$ where $N$ is millions).
- **Quasi-Newton Methods (L-BFGS)**: Approximates inverse Hessian. Works well in full batch deterministic mode, but does not transfer well to mini-batch settings.

## Regularization
Techniques to reduce the gap between training and validation error.

### Early Stopping and Model Ensembles
- **Early Stopping**: Stop training when accuracy on the validation set decreases.
- **Model Ensembles**: 
  1. Train multiple independent models.
  2. At test time, average their results (average predicted probability distributions).
  - Can yield ~2% extra performance.
  - **Trick**: Use multiple snapshots of a single model during training (with cyclic learning rate schedules).
  - **Polyak Averaging**: Keep a moving average of the parameter vector during training and use that at test time.

### L2, L1, and Elastic Net
Regularization term $R(W)$ added to the loss:
- **L2 Regularization (Weight Decay)**: $R(W) = \sum_k \sum_l W_{k,l}^2$
- **L1 Regularization**: $R(W) = \sum_k \sum_l |W_{k,l}|$
- **Elastic Net (L1 + L2)**: $R(W) = \sum_k \sum_l \beta W_{k,l}^2 + |W_{k,l}|$

### Dropout
[KEY CONCEPT] **Dropout**: In each forward pass, randomly set some neurons to zero with probability $p$ (e.g., 0.5).

[CODE]
```python
p = 0.5 # probability of keeping a unit active
def train_step(X):
    # forward pass for 3-layer network
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < p # dropout mask
    H1 *= U1 # drop!
    ...
```
- **Why it works**: Forces the network to have a redundant representation; prevents co-adaptation of features. Can also be viewed as training a large ensemble of subnetworks that share parameters.

**The Scaling Problem:**
Imagine a layer with 100 neurons. During training, if we use dropout with $p=0.5$ (keeping 50% active), only 50 neurons are firing. Let's say each outputs a value of $10$. The total signal sent to the next layer is $50 \times 10 = 500$.
At **test time**, we turn dropout off. All 100 neurons are firing. The total signal sent to the next layer is suddenly $100 \times 10 = 1000$. The next layer will completely malfunction because it was trained to expect a signal of $500$, not $1000$. 

**Vanilla Solution vs. Inverted Dropout:**
- **Vanilla Dropout:** Fixes this by multiplying all activations by $p$ ($0.5$) at *test time* to scale the $1000$ back down to $500$. The annoyance is that this requires adding extra math and changing the code specifically for test time.
- **Inverted Dropout** (The standard practice): Instead of scaling *down* at test time, we scale *up* during training. When we drop 50% of the neurons during training, we immediately divide the remaining survivors by $p$ ($0.5$), which effectively doubles their output. The training signal is artificially boosted to $1000$. At test time, we do absolutely nothing. The 100 neurons naturally output a signal of $1000$, perfectly matching what the network was trained on. 

*Why do this?* It keeps the test-time forward pass as fast and simple as possible.
[CODE]
```python
U1 = (np.random.rand(*H1.shape) < p) / p # scale during training!
H1 *= U1
```

### Data Augmentation
- [KEY CONCEPT] **Data Augmentation**: Transform images during training to artificially increase dataset size and prevent overfitting.
- **Transformations**:
  - Horizontal Flips
  - Random crops and scales
  - Color Jitter (randomize contrast and brightness, or apply PCA-based color offsets).
  - Translation, rotation, stretching, shearing, lens distortions.
- **Test time**: Average predictions over a fixed set of crops (e.g., 10 crops: 4 corners + center, and their flips).

### Other Regularization Patterns
Common pattern: Add random noise during training, marginalize over noise during testing.
- **DropConnect**: Dropout on weights instead of activations.
- **Fractional Max Pooling**: Randomize pooling regions.
- **Stochastic Depth**: Randomly drop entire layers in deep networks (like ResNet) during training.

## Transfer Learning
[KEY CONCEPT] **Transfer Learning**: Using a pre-trained model on a new dataset.
- "You need a lot of data if you want to train/use CNNs" is a myth!
- **Strategy based on data size and similarity**:

| | Very similar dataset | Very different dataset |
|---|---|---|
| **Very little data** | Use Linear Classifier on top layer | Try linear classifier from different stages |
| **Quite a lot of data**| Finetune a few layers | Finetune a larger number of layers |

- Transfer learning is pervasive in object detection (e.g., Fast R-CNN) and image captioning. Word vectors are also typically pre-trained (e.g., word2vec).

## Summary & Key Takeaways
- **Normalization**: Batch Normalization, Layer Normalization, Instance Normalization, Group Normalization.
- **Optimization**: SGD, Momentum, RMSProp, Adam. Adam is a great default.
- **Regularization**: L2, Dropout, Data Augmentation. Add randomness in training, average out in testing.
- **Transfer Learning**: Pre-train on a large dataset (like ImageNet) and fine-tune on your specific task.
- Connection to future: Next lecture covers deep learning software frameworks (TensorFlow, PyTorch, Caffe).
