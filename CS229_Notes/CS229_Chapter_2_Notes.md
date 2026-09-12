# CS229: Chapter 2 - Classification and Logistic Regression

## Overview of Classification
The classification problem is similar to regression, but the values $y$ we want to predict take on a small number of discrete values. 
- **Binary Classification**: $y \in \{0, 1\}$
  - `0`: Negative class (often denoted by `-`)
  - `1`: Positive class (often denoted by `+`)
- $x^{(i)}$ represents the features, and $y^{(i)}$ is the corresponding **label** for the training example.

---

## 2.1 Logistic Regression

Using standard linear regression for classification is problematic because it can output values larger than 1 or smaller than 0, which doesn't make sense when $y \in \{0, 1\}$. 

To fix this, we change the form of our hypothesis $h_\theta(x)$ to use the **logistic function** (or **sigmoid function**):
$$ h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}} $$
where:
$$ g(z) = \frac{1}{1 + e^{-z}} $$

### Properties of the Sigmoid Function
- As $z \to \infty$, $g(z) \to 1$.
- As $z \to -\infty$, $g(z) \to 0$.
- $g(z)$ is always bounded between 0 and 1.
- *Diagram Description*: The plot of $g(z)$ is an S-shaped curve that smoothly increases from 0 to 1, crossing $y = 0.5$ at $z = 0$.

A useful property of the derivative of the sigmoid function, denoted as $g'(z)$:
$$ g'(z) = \frac{d}{dz} \frac{1}{1 + e^{-z}} = \frac{1}{(1 + e^{-z})^2} e^{-z} = \frac{1}{1 + e^{-z}} \left( 1 - \frac{1}{1 + e^{-z}} \right) = g(z)(1 - g(z)) $$

### Probabilistic Assumptions and Maximum Likelihood
We endow our classification model with probabilistic assumptions:
$$ P(y = 1 | x; \theta) = h_\theta(x) $$
$$ P(y = 0 | x; \theta) = 1 - h_\theta(x) $$

This can be written compactly as:
$$ p(y | x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y} $$

Assuming the $n$ training examples are generated independently, the **likelihood** of the parameters is:
$$ L(\theta) = p(\vec{y} | X; \theta) = \prod_{i=1}^n p(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^n (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}} $$

The **log-likelihood**, $\ell(\theta)$, is easier to maximize:
$$ \ell(\theta) = \log L(\theta) = \sum_{i=1}^n y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) \quad \text{--- (2.1)} $$

### Gradient Ascent
To maximize the likelihood, we use **gradient ascent**: 
$$ \theta := \theta + \alpha \nabla_\theta \ell(\theta) $$
*(Note the positive sign because we are maximizing, not minimizing.)*

Working with one training example $(x, y)$, we take derivatives to derive the stochastic gradient ascent rule:
$$ \frac{\partial}{\partial \theta_j} \ell(\theta) = \left( y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)} \right) \frac{\partial}{\partial \theta_j} g(\theta^T x) $$
Using the derivative property $g'(z) = g(z)(1 - g(z))$ and $\frac{\partial}{\partial \theta_j} \theta^T x = x_j$:
$$ \frac{\partial}{\partial \theta_j} \ell(\theta) = (y(1 - g(\theta^T x)) - (1 - y)g(\theta^T x)) x_j = (y - h_\theta(x)) x_j \quad \text{--- (2.2)} $$

This yields the **stochastic gradient ascent rule**:
$$ \theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)} $$
> **Note**: This rule looks identical to the LMS update rule for linear regression! However, it is a different algorithm because $h_\theta(x^{(i)})$ is now a non-linear function of $\theta^T x^{(i)}$. 

### Remark 2.1.1: Alternative Notational Viewpoint
We can define the **logistic loss**:
$$ \ell_{\text{logistic}}(t, y) \triangleq y \log(1 + \exp(-t)) + (1 - y) \log(1 + \exp(t)) \quad \text{--- (2.3)} $$
By plugging in $h_\theta(x) = 1/(1 + e^{-\theta^T x})$, the negative log-likelihood can be rewritten as:
$$ -\ell(\theta) = \ell_{\text{logistic}}(\theta^T x, y) \quad \text{--- (2.4)} $$
Here, $\theta^T x$ or $t$ is often called the **logit**.
Basic calculus gives us:
$$ \frac{\partial \ell_{\text{logistic}}(t, y)}{\partial t} = 1/(1 + \exp(-t)) - y \quad \text{--- (2.5, 2.6)} $$
Using the chain rule, we arrive at the same gradient:
$$ \frac{\partial}{\partial \theta_j} \ell(\theta) = -\frac{\partial \ell_{\text{logistic}}(t, y)}{\partial t} \cdot \frac{\partial t}{\partial \theta_j} = (y - h_\theta(x)) x_j \quad \text{--- (2.7, 2.8)} $$

---

## 2.2 Digression: The Perceptron Learning Algorithm

To force the logistic regression method to output values that are exactly 0 or 1, we can change $g$ to a **threshold function**:
$$ g(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases} $$

If we let $h_\theta(x) = g(\theta^T x)$ with this threshold function and keep the same update rule:
$$ \theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)} $$
we get the **Perceptron Learning Algorithm**.
- Historically (1960s) argued to be a rough model for how individual neurons in the brain work.
- Unlike logistic regression and least squares, it is difficult to give the perceptron meaningful probabilistic interpretations or derive it as a maximum likelihood estimation algorithm.

---

## 2.3 Multi-Class Classification

In multi-class classification, the response variable $y \in \{1, 2, \dots, k\}$.
- Modeled using a **multinomial distribution** with parameters $\phi_1, \dots, \phi_k$ where $\sum_{i=1}^k \phi_i = 1$.
- We introduce $k$ groups of parameters $\theta_1, \dots, \theta_k$, each being a vector in $\mathbb{R}^d$.

To turn the outputs $(\theta_1^T x, \dots, \theta_k^T x)$ into a valid probability vector with nonnegative entries that sum to 1, we use the **softmax function**:
$$ \text{softmax}(t_1, \dots, t_k) = \begin{bmatrix} \frac{\exp(t_1)}{\sum_{j=1}^k \exp(t_j)} \\ \vdots \\ \frac{\exp(t_k)}{\sum_{j=1}^k \exp(t_j)} \end{bmatrix} \quad \text{--- (2.9)} $$
The inputs $t$ to the softmax function are often called **logits**.

The probabilistic model becomes:
$$ P(y = i | x; \theta) = \phi_i = \frac{\exp(\theta_i^T x)}{\sum_{j=1}^k \exp(\theta_j^T x)} \quad \text{--- (2.11)} $$

### Cross-Entropy Loss
The negative log-likelihood of a single example $(x, y)$ is:
$$ -\log p(y | x, \theta) = -\log \left( \frac{\exp(\theta_y^T x)}{\sum_{j=1}^k \exp(\theta_j^T x)} \right) \quad \text{--- (2.12)} $$
Thus, the negative log-likelihood of the training data (loss function) is:
$$ \ell(\theta) = \sum_{i=1}^n -\log \left( \frac{\exp(\theta_{y^{(i)}}^T x^{(i)})}{\sum_{j=1}^k \exp(\theta_j^T x^{(i)})} \right) \quad \text{--- (2.13)} $$

This is modularized into the **cross-entropy loss**:
$$ \ell_{ce}((t_1, \dots, t_k), y) = -\log \left( \frac{\exp(t_y)}{\sum_{j=1}^k \exp(t_j)} \right) \quad \text{--- (2.14)} $$
The gradient of the cross-entropy loss with respect to $t_i$:
$$ \frac{\partial \ell_{ce}(t, y)}{\partial t_i} = \phi_i - 1\{y = i\} \quad \text{--- (2.16)} $$
Where $1\{\cdot\}$ is the indicator function. In vectorized notation, this is:
$$ \frac{\partial \ell_{ce}(t, y)}{\partial t} = \phi - e_y \quad \text{--- (2.17)} $$
Using the chain rule, the gradient of the loss with respect to the parameter $\theta_i$ is:
$$ \frac{\partial \ell(\theta)}{\partial \theta_i} = \sum_{j=1}^n (\phi_i^{(j)} - 1\{y^{(j)} = i\}) \cdot x^{(j)} \quad \text{--- (2.19)} $$
With these gradients, one can implement (stochastic) gradient descent to minimize $\ell(\theta)$.

---

## 2.4 Another Algorithm for Maximizing $\ell(\theta)$: Newton's Method

**Newton's Method** is an algorithm for finding the zeroes of a function $f: \mathbb{R} \to \mathbb{R}$ such that $f(\theta) = 0$.
The update rule is:
$$ \theta := \theta - \frac{f(\theta)}{f'(\theta)} $$
- *Diagram Description*: Newton's method approximates the function $f$ via a linear function that is tangent to $f$ at the current guess $\theta$. It solves for where this tangent line equals zero (intersects the x-axis) and sets that point as the next guess for $\theta$, yielding rapid convergence.

To maximize $\ell(\theta)$, we want to find where its first derivative is zero. By setting $f(\theta) = \ell'(\theta)$, we apply Newton's method:
$$ \theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)} $$

### Multidimensional Setting (Newton-Raphson Method)
Since $\theta$ is vector-valued in logistic regression, we generalize the method:
$$ \theta := \theta - H^{-1} \nabla_\theta \ell(\theta) $$
where $\nabla_\theta \ell(\theta)$ is the vector of partial derivatives and $H$ is the **Hessian** matrix (an $d \times d$ or $(d+1) \times (d+1)$ matrix if including the intercept):
$$ H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j} $$

**Newton's Method vs. Gradient Descent:**
- **Pros**: Enjoys much faster convergence and requires many fewer iterations to get very close to the minimum.
- **Cons**: One iteration is more expensive because it requires finding and inverting an $d \times d$ Hessian matrix.
- When applied to maximize the logistic regression log-likelihood function, this resulting method is also called **Fisher scoring**.