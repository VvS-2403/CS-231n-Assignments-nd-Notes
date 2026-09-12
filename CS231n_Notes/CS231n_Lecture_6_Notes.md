# CS231n Lecture 6: Training Neural Networks, Part I
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Administrative](#administrative)
- [Where we are now...](#where-we-are-now)
  - [Computational graphs](#computational-graphs)
  - [Neural Networks](#neural-networks)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
  - [Convolutional Layer](#convolutional-layer)
  - [Learning network parameters through optimization](#learning-network-parameters-through-optimization)
  - [Mini-batch SGD](#mini-batch-sgd)
- [Part 1 Overview](#part-1-overview)
- [Activation Functions](#activation-functions)
  - [Sigmoid](#sigmoid)
  - [tanh(x)](#tanhx)
  - [ReLU (Rectified Linear Unit)](#relu-rectified-linear-unit)
  - [Leaky ReLU & PReLU](#leaky-relu--prelu)
  - [Exponential Linear Units (ELU)](#exponential-linear-units-elu)
  - [Maxout "Neuron"](#maxout-neuron)
  - [Activation Functions Summary](#activation-functions-summary)
- [Data Preprocessing](#data-preprocessing)
- [Weight Initialization](#weight-initialization)
  - [Constant Initialization](#constant-initialization)
  - [Small Random Numbers](#small-random-numbers)
  - [Xavier Initialization](#xavier-initialization)
  - [He et al. Initialization](#he-et-al-initialization)
- [Batch Normalization](#batch-normalization)
- [Babysitting the Learning Process](#babysitting-the-learning-process)
  - [Step 1: Preprocess the data](#step-1-preprocess-the-data)
  - [Step 2: Choose the architecture](#step-2-choose-the-architecture)
  - [Double check that the loss is reasonable](#double-check-that-the-loss-is-reasonable)
  - [Lets try to train now...](#lets-try-to-train-now)
- [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Cross-validation strategy](#cross-validation-strategy)
  - [Random Search vs. Grid Search](#random-search-vs-grid-search)
  - [Hyperparameters to play with](#hyperparameters-to-play-with)
  - [Monitoring learning](#monitoring-learning)
- [Summary & Key Takeaways](#summary--key-takeaways)
- [Next time: Training Neural Networks, Part 2](#next-time-training-neural-networks-part-2)

---

## Administrative

- **Assignment 1** was due yesterday.
- **Assignment 2** is out, due Wed May 2. Q5 will be released in a few days.
- **Project proposal** due Wed April 25.

---

## Where we are now...

### Computational graphs
[DIAGRAM] An illustration of a computational graph for a linear classifier with hinge loss and regularization. Nodes represent operations (matrix multiplication, hinge loss calculation, regularization), and edges show the flow of data ($x$, $W$, scores $s$, and loss $L$).
- Loss function: $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$
- Score function: $f = Wx$

### Neural Networks
[DIAGRAM] A 2-layer Neural Network showing input $x$ ($3072$), weights $W_1$, hidden layer $h$ ($100$), weights $W_2$, and scores $s$ ($10$). Below it, CIFAR-10 images are shown.
- **Linear score function**: $f = Wx$
- **2-layer Neural Network**: $f = W_2 \max(0, W_1 x)$

### Convolutional Neural Networks
[DIAGRAM] Illustration of a Convolutional Neural Network from LeCun et al. 1998, showing input image mapping to multiple "Image Maps" via convolutions, followed by subsampling, and then fully connected layers to generate the final output.

### Convolutional Layer
[DIAGRAM] A $32 \times 32 \times 3$ image is convolved with a $5 \times 5 \times 3$ filter. A single neuron looks at a local spatial region but through the full depth of the volume. By sliding this filter over all spatial locations, it produces a single $28 \times 28$ **activation map**.
- If we apply 6 separate $5 \times 5$ filters, we get 6 separate activation maps.
- We stack these to get a "new image" of size $28 \times 28 \times 6$.

**Parameter Sharing:**
- **Concept:** If a feature (like an edge) is useful to compute at one spatial position, it is likely useful at another. Instead of learning separate weights for every location, a single filter (kernel) is "shared" and slid across the entire image.
- **Advantages:**
  1. **Drastic parameter reduction:** Instead of $(N \times M) \times H$ weights (fully connected), a $K \times K$ filter only needs $K^2$ weights.
  2. **Translation Equivariance:** The network learns to find a pattern no matter where it appears in the image.

### Learning network parameters through optimization
[DIAGRAM] A landscape image representing the loss function. A path descends through the landscape towards a minimum, illustrating gradient descent.

[CODE]
```python
# Vanilla Gradient Descent
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size * weights_grad # perform parameter update
```

### Mini-batch SGD
**Loop:**
1. **Sample** a batch of data.
2. **Forward prop** it through the graph (network), get loss.
3. **Backprop** to calculate the gradients.
4. **Update** the parameters using the gradient.

---

## Part 1 Overview
1. **One time setup**: activation functions, preprocessing, weight initialization, regularization, gradient checking.
2. **Training dynamics**: babysitting the learning process, parameter updates, hyperparameter optimization.
3. **Evaluation**: model ensembles.

---

## Activation Functions

### Sigmoid
[FORMULA] $\sigma(x) = \frac{1}{1 + e^{-x}}$
[DIAGRAM] A plot of the Sigmoid function, which squashes input numbers to the range $[0, 1]$.

- Squashes numbers to range $[0, 1]$
- Historically popular since they have nice interpretation as a saturating "firing rate" of a neuron.

**3 problems:**
1. **Saturated neurons "kill" the gradients**: When $x = -10$ or $x = 10$, the local gradient $\frac{\partial \sigma}{\partial x}$ is nearly 0. During backpropagation, multiplying the upstream gradient by 0 effectively kills the gradient flowing through this node.
2. **Sigmoid outputs are not zero-centered**: If inputs to a neuron are always positive, the gradients on the weights $w$ during backpropagation will always be all positive or all negative. This causes a zig-zag path in the weight updates, which is inefficient.
   - **Why?** The gradient is $\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot x_i$. If $x_i > 0$ always, the sign of the gradient is entirely determined by the upstream gradient $\frac{\partial L}{\partial z}$.
   - **Result:** All weights for a given neuron are forced to update in the exact same direction (all increase or all decrease). To move in a $(+,-)$ direction, the optimizer must take alternating $(+,+)$ and $(-,-)$ steps, causing an inefficient zigzag. **Mini-batches** and **Momentum** help mitigate this by averaging these erratic updates out.
3. **exp() is a bit compute expensive**.

### tanh(x)
[DIAGRAM] A plot of the tanh function, which squashes input numbers to the range $[-1, 1]$.

- Squashes numbers to range $[-1, 1]$.
- **Zero centered** (nice!).
- **Still kills gradients** when saturated :(
  - **Saturation:** When the input $x$ is very large (positive) or very small (negative), the `tanh` function outputs values extremely close to $1$ or $-1$. In these "flat" regions, the local gradient (slope) $\frac{\partial}{\partial x} \tanh(x)$ is practically $0$. 
  - During backpropagation, multiplying by this $0$ local gradient stops the flow of error backwards, preventing weights from updating.

### ReLU (Rectified Linear Unit)
[FORMULA] $f(x) = \max(0, x)$
[DIAGRAM] A plot of the ReLU function. It is 0 for $x < 0$ and linear with slope 1 for $x \ge 0$.

- **Does not saturate** (in + region).
- **Very computationally efficient**.
- **Converges much faster** than sigmoid/tanh in practice (e.g. 6x).
- Actually more biologically plausible than sigmoid.

**Problems:**
- **Not zero-centered output**.
- **An annoyance**: What is the gradient when $x < 0$? The gradient is 0.
- [KEY CONCEPT] **Dead ReLU**: A ReLU neuron can "die" and never activate again if a large weight update pushes it off the data cloud such that its input is always negative. It will never update its weights.
> **Key Insight**: People like to initialize ReLU neurons with slightly positive biases (e.g., 0.01) to increase the chance of them being active at initialization and receiving gradients.

### Leaky ReLU & PReLU
**Leaky ReLU:**
[FORMULA] $f(x) = \max(0.01x, x)$
[DIAGRAM] Plot of Leaky ReLU. For $x < 0$, it has a small positive slope instead of being flat.
- Does not saturate.
- Computationally efficient.
- Converges much faster than sigmoid/tanh in practice.
- **Will not "die"**.

**Parametric Rectifier (PReLU):**
[FORMULA] $f(x) = \max(\alpha x, x)$
- Instead of hard-coding the negative slope $\alpha$ (like $0.01$ in Leaky ReLU), $\alpha$ is treated as a **learnable parameter**. 
- The neural network uses backpropagation to learn the optimal slope for the negative region on its own. 
- Usually, there is one learnable $\alpha$ per layer or one per feature channel, giving the network more flexibility without requiring us to guess the perfect hyperparameter.

### Exponential Linear Units (ELU)
[FORMULA] 
$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(\exp(x) - 1) & \text{if } x \le 0 \end{cases}$
[DIAGRAM] Plot of ELU function. Similar to ReLU for $x > 0$, but smoothly curves to $-\alpha$ for $x \le 0$.

- All benefits of ReLU (no positive saturation, fast convergence).
- **Closer to zero mean outputs**: Helps alleviate the zigzag path problem.
- **Smoothness**: Unlike Leaky/PReLU which have a sharp "kink" at $x=0$, ELU smoothly transitions into the negative values. This continuous gradient makes optimization more stable.
- **Robustness to noise**: Unlike Leaky ReLU which shoots down to negative infinity, ELU bottoms out at $-\alpha$. This "negative saturation" prevents massive negative outliers from wreaking havoc on the network.
- **Computation requires exp()**: Calculating $e^x$ is strictly slower and more computationally expensive than the simple multiplication used in Leaky ReLU/PReLU.

### Maxout "Neuron"
[FORMULA] $f(x) = \max(w_1^T x + b_1, w_2^T x + b_2)$

**How it works:** 
Normally, a neuron calculates a dot product ($w^T x + b$) and passes it through an activation function like ReLU. A Maxout neuron does not use a traditional activation function. Instead, it has **two complete sets of weights and biases**. It calculates the dot product for both sets, and simply outputs the maximum of the two.

**Pros:**
- **Generalizes ReLU and Leaky ReLU:** Standard ReLU is just a special case of Maxout where one set of weights is forced to be zero ($\max(0, w^T x + b)$). 
- **No Saturation & No Dying:** Because it is just composed of two linear equations, it never saturates (unlike Sigmoid) and it does not suffer from the "dead neuron" problem (unlike ReLU), because if one linear function becomes negative, the other can take over and still pass gradients.

**Cons:**
- **Doubles the parameters:** Because every single neuron requires two sets of weights ($w_1$ and $w_2$), it literally doubles the parameter count of the layer. This makes the network heavier, slower, and more prone to overfitting.

### Activation Functions Summary
> **Key Insight**: In practice:
> - Use **ReLU**. Be careful with your learning rates.
> - Try out Leaky ReLU / Maxout / ELU.
> - Try out tanh but don't expect much.
> - **Don't use sigmoid**.

---

## Data Preprocessing

### Step 1: Preprocess the data
[DIAGRAM] Scatter plots showing original data, zero-centered data (mean subtracted), and normalized data (divided by standard deviation).

[CODE]
```python
# Zero-center
X -= np.mean(X, axis=0)

# Normalize
X /= np.std(X, axis=0)
```
- In practice, you may also see **PCA** and **Whitening** of the data.

> **Key Insight**: **TLDR for Images:** Center only.
> - Subtract the mean image (e.g., AlexNet). Mean image is a $[32, 32, 3]$ array.
> - Subtract per-channel mean (e.g., VGGNet). Mean along each channel = 3 numbers.
> - It is not common to normalize variance, to do PCA, or whitening for images.

### Why Zero-Centering Matters Everywhere
You will notice a recurring theme in neural networks: we desperately want our data, our weights, and our activations to be **centered around zero**. Here is why it is so critical across the board:

1. **For Data Inputs:** If your raw input data (like pixels) is strictly positive, the first layer of your network suffers from the "all-or-nothing" gradient problem. The weights in the first layer will be forced into an inefficient zigzag optimization path because they can only all increase or all decrease together. Subtracting the mean (zero-centering) fixes this immediately.
2. **For Hidden Layer Activations:** Just as positive inputs hurt the first layer, positive activations (like those from a standard ReLU) hurt the *next* layer. If Layer 1 only outputs positive numbers, Layer 2 is forced into the zigzag path. This is why zero-centered activations (like `Tanh` or ELU) and techniques like **Batch Normalization** are so powerful—they re-center the data to zero midway through the network.
3. **For Weight Initialization:** We initialize weights with a mean of 0 (e.g., $\mathcal{N}(0, \sigma^2)$) so the network starts completely unbiased. If we started with a positive mean, the network would have a massive initial bias, and multiplying positive inputs by positive weights layer after layer would cause the signal's mean to explode away from zero as it travels deep into the network.

---

## Weight Initialization

### Constant Initialization
- **Q**: What happens when $W = \text{constant}$ init is used?
- **A**: If all neurons have the same weights, they will compute the same gradients during backpropagation and will undergo the exact same parameter updates. There will be no symmetry breaking.

### Small Random Numbers
- **First idea**: Small random numbers (Gaussian with zero mean and 1e-2 standard deviation).
[CODE]
```python
W = 0.01 * np.random.randn(D, H)
```
- Works ~okay for small networks, but problems with deeper networks.
- [DIAGRAM] Histograms of activations across 10 layers. If weights are initialized too small, **all activations become zero** in deeper layers. During the backward pass, local gradients ($X$) are zero, so gradients flowing back will also become zero.

- What if we use a larger multiplier, e.g., `1.0 * np.random.randn(fan_in, fan_out)`?
- **Result**: Almost all neurons completely saturated (for tanh, either -1 or 1). Gradients will be all zero.

### Xavier Initialization
**Intuition:** The goal is to keep the variance of the signal constant across layers so it doesn't explode or vanish. If you add $n$ inputs together, variance naturally grows by a factor of $n$. Xavier stops this by scaling weights down proportionally.
**Normal Distribution Variant:**
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$
- **Why $\frac{2}{n_{in} + n_{out}}$? The Compromise:**
  - **Forward Pass:** To keep the signal from vanishing as it moves *forward*, the math dictates the weight variance must be $\frac{1}{n_{in}}$.
  - **Backward Pass:** When gradients flow *backward* during backpropagation, the layer effectively acts in reverse, receiving $n_{out}$ inputs. To keep gradients from vanishing, the math dictates the weight variance must be $\frac{1}{n_{out}}$.
  - **The Solution:** Because we can't satisfy both perfectly (unless $n_{in} = n_{out}$), Xavier Glorot took the **harmonic mean** of the two ideal variances to strike a perfect compromise: $\frac{2}{n_{in} + n_{out}}$.
*(Note: A simpler, commonly used version only uses the fan-in: $W \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right)$)*

**Uniform Distribution Variant:**
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \ \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

[CODE]
```python
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
```
- Reasonable initialization. Mathematical derivation assumes linear activations.
- **Mathematical Guarantee:** It forces the variance of the outputs to be perfectly equal to the variance of its inputs ($\text{Var}(y) = \text{Var}(x)$).
- [DIAGRAM] Histograms show that activation distributions remain **well-behaved** across layers.
  - **What does "well-behaved" mean here?** If weights are too small, the variance shrinks by a fraction at every layer (e.g., $0.5 \times 0.5 \times \dots$), causing the activations to eventually collapse to $0$ (vanishing). If weights are too big, variance multiplies exponentially (e.g., $2 \times 2 \times \dots$), causing activations to explode. Because Xavier makes the multiplier exactly $1$, the "spread" of the numbers stays identical from Layer 1 all the way to Layer 50.
- **Problem**: When using the **ReLU** nonlinearity, Xavier initialization breaks because ReLU zeroes out half the inputs, halving the variance.

### He et al. Initialization
**Intuition:** Xavier's math assumes a linear, symmetric activation function. ReLU breaks this because it zeroes out half the signal, halving the variance. He initialization fixes this by simply doubling the variance of the weights to compensate for the lost signal.
**Normal Distribution Variant:**
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

**Uniform Distribution Variant:**
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \ \sqrt{\frac{6}{n_{in}}}\right)$$

[CODE]
```python
W = np.random.randn(fan_in, fan_out) / np.sqrt(2 / fan_in)
```
- Developed by He et al., 2015.
- Accounts for the ReLU nonlinearity by adding a `2` in the numerator, compensating for the variance reduction.

---

## Batch Normalization

**The Core Concepts:**
Before understanding *why* it works, it is crucial to understand exactly *what* is being normalized:
- **What is a "Layer"?** A layer is a collection of neurons (like a Fully Connected or Convolutional layer). Each neuron computes a weighted sum of its inputs, resulting in a raw numerical output (often called $z$).
- **What are we normalizing?** We are normalizing those raw $z$ values, usually *immediately before* they are passed into the activation function (like ReLU). Importantly, we normalize the outputs of **each neuron independently**.
- **What is a "Batch"?** During training, we process data in "mini-batches" (e.g., 32 images at a time) rather than one by one. This means a single neuron will output 32 different $z$ values simultaneously (one for each image). Batch Normalization calculates the mean and variance across those 32 numbers, and uses them to standardize that specific neuron's outputs.

### The Problem: Internal Covariate Shift
To understand Batch Normalization, you must understand the exact problem it was invented to solve:
- **Covariate Shift:** In statistics, this occurs when the distribution of your input data changes. For example, if you train a model to recognize dogs using photos taken in bright daylight, but test it on photos taken at night, the model will struggle. The underlying task is the same, but the distribution of the inputs (the covariates) has shifted.
- **The "Internal" Problem:** A deep neural network is essentially a chain of models. Layer 2 treats the outputs of Layer 1 as its inputs. During training, Layer 1 is constantly updating its weights via gradient descent. Because Layer 1's weights change at every step, the mathematical distribution (the scale and center) of the numbers it outputs also constantly changes.
- **The Result:** Layer 2 suffers from covariate shift *during* training. Every time Layer 2 tries to learn how to process its inputs, Layer 1 alters the distribution of those inputs. Deep layers are forced to constantly waste time readjusting their weights just to keep up with the moving targets set by earlier layers. This compounds as you go deeper, severely slowing down the entire network.

### The Solution: Batch Normalization
Batch Normalization solves this by acting as a stabilizer between layers:
1. **The Standardization:** It takes the batch of numbers output by a layer and forcibly standardizes them (subtracts the mean, divides by variance) so they always have a mean of $0$ and a variance of $1$. This guarantees that the next layer always receives data in a consistent, stable range, completely removing the chaotic shifting caused by earlier layers.
2. **The Learnable Parameters ($\gamma$ and $\beta$):** Strictly forcing a mean of $0$ and variance of $1$ can be too restrictive (e.g., it might prevent the network from using the useful, non-linear parts of an activation function). Therefore, BN introduces two learnable parameters: $\gamma$ (to scale the variance) and $\beta$ (to shift the mean). This allows the network to learn the exact optimal mean and variance it needs, but it learns it from a clean, mathematically stable baseline.

### Decorrelated Batch Normalization (Whitening)
Standard Batch Normalization normalizes the output of **each neuron independently**. While this ensures each feature has a mean of 0 and variance of 1, it completely ignores the relationships *between* different neurons. 

**The Problem with Standard BN:** 
If Neuron A and Neuron B learn to detect very similar patterns, their outputs will be highly correlated. This is redundant; the network is wasting capacity. Standard BN does nothing to fix this correlation.

**What Decorrelated BN Does:**
Decorrelated Batch Normalization (also related to ZCA Whitening) goes a step further. Instead of just dividing by the variance of individual neurons, it calculates the full **covariance matrix** of all the features in the layer. It then transforms the data so that:
1. Every neuron has a mean of 0 and variance of 1.
2. The correlation between any two distinct neurons is exactly **0** (the features become orthogonal).

**Why it's good:** It forces the network to learn completely distinct, non-redundant features, which can lead to better generalization and optimization.
**Why it's rare in practice:** Calculating the inverse square root of a full covariance matrix ($\Sigma^{-1/2}$) requires complex matrix decompositions (like SVD or Eigendecomposition). Doing this for every mini-batch at every layer is massively computationally expensive compared to the simple scalar division of standard BN.

"You want zero-mean unit-variance activations? Just make them so." - Ioffe and Szegedy, 2015

Consider a batch of activations at some layer. To make each dimension zero-mean unit-variance, apply:
[FORMULA] 
$$\widehat{x}^{(k)} = \frac{x^{(k)} - \text{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}$$
This is a vanilla differentiable function.

**Algorithm:**
1. Compute the empirical mean and variance independently for each dimension across the mini-batch.
2. Normalize.
3. Allow the network to squash the range if it wants to:
[FORMULA]
$$y^{(k)} = \gamma^{(k)} \widehat{x}^{(k)} + \beta^{(k)}$$
Note: The network can learn $\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$ and $\beta^{(k)} = \text{E}[x^{(k)}]$ to recover the identity mapping if that is optimal.

**Properties:**
- Usually inserted after Fully Connected or Convolutional layers, and before the nonlinearity.
- Improves gradient flow through the network.
- Allows higher learning rates.
- Reduces the strong dependence on initialization.
- Acts as a form of regularization in a funny way, and slightly reduces the need for dropout.
- **Test time**: The mean and standard deviation are not computed based on the batch. Instead, a single fixed empirical mean and variance of activations, estimated via running averages during training, is used.

---

## Babysitting the Learning Process

### Step 1: Preprocess the data
### Step 2: Choose the architecture
- E.g., one hidden layer of 50 neurons.

### Double check that the loss is reasonable
[CODE]
```python
# Initialize a model
model = init_two_layer_model(32*32*3, 50, 10) 
# disable regularization
loss, grad = two_layer_net(X_train, model, y_train, 0.0)
print loss 
# Output: ~2.3
```
- With 10 classes and no regularization, expected initial Softmax loss is $-\log(0.1) \approx 2.3$. This is a sanity check!
- If you **crank up regularization**, the loss should go up.

### Lets try to train now...
> **Tip**: Make sure that you can overfit a very small portion of the training data.
- Take 20 examples from CIFAR-10, turn off regularization, use vanilla SGD.
- The loss should go to 0, and training accuracy should hit 1.00.

**Finding the learning rate:**
- Start with small regularization and find a learning rate that makes the loss go down.
- If loss is not going down: learning rate is **too low**.
- If loss explodes to NaN: learning rate is **too high**.
- **Tip**: Rough range for learning rate cross-validation is usually $[1e-3, \dots, 1e-5]$.

---

## Hyperparameter Optimization

### Cross-validation strategy
**Coarse -> Fine** cross-validation in stages.
- **First stage (Coarse)**: Only a few epochs to get a rough idea of what parameters work.
- **Second stage (Fine)**: Longer running time, finer search around the best values from the coarse stage.

> **Tip for detecting explosions**: If the cost is ever $> 3 \times$ the original cost, break out early to save time.

### Random Search vs. Grid Search
[DIAGRAM] Comparison of Grid Layout vs. Random Layout for sampling hyperparameters.
- **Random Search** is preferred. It is better to optimize in log space (e.g., `10 ** uniform(-5, 5)`).
- Why Random Search? Some parameters are much more important than others (Bergstra and Bengio, 2012). Random search samples more distinct values for the important parameters, leading to better optimization.

### Hyperparameters to play with
- Network architecture
- Learning rate, its decay schedule, update type
- Regularization (L2, Dropout strength)

### Monitoring learning
**1. Monitor and visualize the loss curve**
- [DIAGRAM] A plot showing loss over epochs.
- A very high learning rate causes loss to explode.
- A high learning rate causes loss to drop quickly but plateau at a high value.
- A low learning rate causes slow linear decay.
- A good learning rate causes a smooth curve that settles at a low value.
- The "width" of the noisy curve can indicate batch size effects.

**2. Monitor and visualize the accuracy**
- [DIAGRAM] A plot showing training and validation accuracy over time.
- **Big gap** = overfitting $\rightarrow$ increase regularization strength.
- **No gap** = underfitting $\rightarrow$ increase model capacity.

**3. Track the ratio of weight updates / weight magnitudes**
- Calculate the L2 norm of the update and divide by the L2 norm of the parameter vector.
- You want this ratio to be somewhere around $0.001$ or so ($10^{-3}$).

---

## Summary & Key Takeaways
We looked in detail at:
- **Activation Functions**: Use ReLU.
- **Data Preprocessing**: Images: subtract mean.
- **Weight Initialization**: Use Xavier or He et al. initialization.
- **Batch Normalization**: Use it! It makes training much easier.
- **Babysitting the Learning process**: Sanity check initial loss, overfit a small batch.
- **Hyperparameter Optimization**: Use random sampling, sample hyperparameters in log space when appropriate.

---

## Next time: Training Neural Networks, Part 2
- Parameter update schemes
- Learning rate schedules
- Gradient checking
- Regularization (Dropout etc.)
- Evaluation (Ensembles etc.)
- Transfer learning / fine-tuning
