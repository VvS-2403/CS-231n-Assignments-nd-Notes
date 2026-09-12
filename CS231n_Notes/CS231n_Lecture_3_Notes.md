# CS231n Lecture 3: Loss Functions and Optimization

**Stanford CS231n | Spring 2018**

---

## Table of Contents
- [Introduction](#introduction)
- [Linear Classification & Score Functions](#linear-classification--score-functions)
- [Loss Functions](#loss-functions)
  - [Multiclass SVM Loss](#multiclass-svm-loss)
  - [Softmax Loss (Cross-Entropy)](#softmax-loss-cross-entropy)
- [Visualizing the Loss Function](#visualizing-the-loss-function)
- [Optimization Strategies](#optimization-strategies)
- [Computing the Gradient](#computing-the-gradient)
- [Gradient Descent](#gradient-descent)

---

## Introduction
In image classification, we deal with three core components:
1. **Score Function:** A parameterized mathematical function mapping raw image pixels to class scores.
2. **Loss Function:** A metric that measures how well the predicted scores align with the ground truth labels.
3. **Optimization:** The algorithmic process of finding the exact parameters (weights) that minimize this loss function.

---

## Linear Classification & Score Functions
The simplest score function is a linear mapping. For an image $x_i$ (flattened to a $D$-dimensional vector) and a weight matrix $W$:

$$ f(x_i, W) = W x_i + b $$

- **$W$ (Weights):** Matrix of size $[K \times D]$ ($K$ classes, $D$ pixels). Each row acts as a **template** for a specific class. The dot product measures how well the image matches the template.
- **$b$ (Bias):** Vector of size $[K \times 1]$. It represents data-independent preferences (e.g., if cats are highly frequent in the dataset, the cat bias will be higher).

---

## Loss Functions
The loss function $L$ quantifies our "unhappiness" with the current weights. High loss means bad predictions; low loss means good predictions. 

### Multiclass SVM Loss
The SVM loss wants the correct class score to be higher than all incorrect class scores by a fixed, safe margin $\Delta$ (usually set to 1).

For a single example $(x_i, y_i)$, the loss $L_i$ is:
$$ L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta) $$

- $s_j$ is the predicted score for the $j$-th class.
- $s_{y_i}$ is the predicted score for the **correct** class.
- The $\max(0, -)$ operation is called the **hinge loss**. It clamps negative values at zero. If the correct class score is greater than the incorrect score by at least $\Delta$, the loss is zero.
- **Data Loss vs Regularization Loss:** The total loss $L$ averages $L_i$ over all $N$ examples and adds a regularization penalty $R(W)$ to prevent the weights from becoming too complex and overfitting the training data.
  $$ L = \frac{1}{N} \sum_i L_i + \lambda R(W) $$

### Softmax Loss (Cross-Entropy)
Instead of raw margins, Softmax treats the scores as unnormalized log probabilities. It exponentiates them and normalizes them to form a valid probability distribution:
$$ P(Y=k | X=x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}} $$

The loss is the negative log likelihood of the correct class:
$$ L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right) $$

**SVM vs. Softmax:**
- **SVM:** Operates locally. Once the correct class beats an incorrect class by $\Delta$, the loss becomes exactly 0. It doesn't care if the margin is exactly $\Delta$ or $100 \Delta$.
- **Softmax:** Operates globally. It is *never* fully satisfied. It constantly pushes the correct class probability toward 1 and incorrect probabilities toward 0, no matter how good the scores currently are.

---

## Visualizing the Loss Function
Because weight matrices are incredibly high-dimensional (e.g., 30,730 dimensions for CIFAR-10), we cannot easily visualize the loss landscape. However, we can slice through this space:
- By picking a random direction $W_1$ and graphing $L(W + a W_1)$, we generate 1D slices.
- By picking two random directions and graphing $L(W + a W_1 + b W_2)$, we get 2D heatmaps.

**Observations for SVM:**
- The hinge loss creates a **piecewise-linear** bowl shape structure.
- The loss function is **convex**, meaning it has a single global minimum (for linear classifiers). When we later transition to Neural Networks, these terrains will become highly non-convex, bumpy landscapes.

*(Note on non-differentiability: The kinks at the bottom of the hinge loss are technically non-differentiable, but in practice, we use subgradients and step over them).*

---

## Optimization Strategies
Optimization is the process of navigating the high-dimensional loss landscape to find the lowest point.

### Strategy 1: Random Search (Very Bad Idea)
- Try thousands of randomly generated weight matrices and save the one with the lowest loss.
- **Result:** ~15.5% accuracy on CIFAR-10 (barely better than random guessing).

### Strategy 2: Random Local Search (Bad Idea)
- Start with random weights. Take a small random step. If the loss decreases, keep the step; if it increases, discard it.
- **Result:** ~21.4% accuracy. Highly inefficient and wasteful.

### Strategy 3: Following the Gradient (The Good Idea)
Instead of guessing randomly, we can mathematically compute the direction of steepest descent. This direction is given by the **negative gradient** of the loss function. 
- Analogy: Hiking blindfolded down a mountain by feeling the slope with your feet and stepping in the steepest downward direction.

---

## Computing the Gradient
In multiple dimensions, the gradient is a vector of partial derivatives, representing the slope of the loss function along each dimension of the weight matrix.

### Numerical Gradient
Compute the gradient using the finite difference approximation:
$$ \frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} $$
- **Pros:** Very easy to write and mathematically robust.
- **Cons:** Incredibly slow. You must evaluate the loss function for every single parameter in $W$ individually. It is also an approximation due to floating-point constraints on $h$.

### Analytical Gradient
Compute the exact formula for the derivative using calculus.
- **Pros:** Extremely fast and mathematically exact.
- **Cons:** Error-prone to derive and implement.
- **Best Practice (Gradient Check):** Always use the analytical gradient in production, but verify its correctness against the numerical gradient during debugging.

---

## Gradient Descent
The algorithm that continuously evaluates the gradient and takes a step in the negative direction.

```python
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size * weights_grad # Parameter update
```
- **Step Size (Learning Rate):** The most critical hyperparameter. If it's too small, convergence is agonizingly slow. If it's too large, you will overshoot the minimum and the loss will explode.

### Stochastic Gradient Descent (SGD)
Evaluating the gradient over the entire dataset ($N$ in the millions) is computationally impossible for every single step. 
- **Minibatches:** Instead, we evaluate the gradient on small batches of data (e.g., 32, 64, 128, 256 examples) at a time.
- Because datasets contain highly correlated examples, a minibatch gives a slightly noisy but highly accurate estimate of the true full-batch gradient.
- This results in significantly faster, more frequent weight updates.