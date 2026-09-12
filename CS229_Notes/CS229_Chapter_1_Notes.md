# Chapter 1: Linear Regression

## Introduction to the Problem
To motivate the problem, consider a dataset predicting housing prices based on features like living area and the number of bedrooms.
- Let $x^{(i)}$ denote the input variables (features) for the $i$-th training example. For instance, $x^{(i)}_1$ is the living area, and $x^{(i)}_2$ is the number of bedrooms.
- The input vectors are generally multi-dimensional, $x \in \mathbb{R}^d$.
- We define $y^{(i)}$ as the target variable (e.g., house price) for the $i$-th example.
- A pair $(x^{(i)}, y^{(i)})$ is called a training example, and a dataset of $n$ examples is the training set.

To perform supervised learning, we represent functions/hypotheses $h$ in a computer. As an initial choice, we approximate $y$ as a linear function of $x$:
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2$$

Here, the $\theta_i$'s are the **parameters** (or weights) parameterizing the space of linear functions mapping from $X$ to $Y$. By convention, we set $x_0 = 1$ (the intercept term), allowing us to write the hypothesis compactly using vectors:
$$h(x) = \sum_{i=0}^d \theta_i x_i = \theta^T x$$
where $d$ is the number of input variables (excluding $x_0$).

### Cost Function
To learn the parameters $\theta$, we define a function that measures how close the predictions $h(x^{(i)})$ are to the true values $y^{(i)}$. We use the ordinary least squares **cost function**:
$$J(\theta) = \frac{1}{2} \sum_{i=1}^n \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

---

## 1.1 LMS Algorithm (Least Mean Squares)

To minimize $J(\theta)$, we can use an iterative search algorithm. **Gradient descent** starts with an initial guess for $\theta$ and repeatedly takes a step in the direction of the steepest decrease of $J$:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$
(This update is simultaneously performed for all $j = 0, \ldots, d$).
Here, $\alpha$ is the **learning rate**.

### Derivation for a Single Example
For a single training example $(x, y)$, the partial derivative is:
$$
\begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_\theta(x) - y)^2 \\
&= 2 \cdot \frac{1}{2} (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x) - y) \\
&= (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} \left( \sum_{i=0}^d \theta_i x_i - y \right) \\
&= (h_\theta(x) - y) x_j
\end{aligned}
$$

For a single example, this yields the **LMS update rule** (or Widrow-Hoff learning rule):
$$\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}_j$$
*Intuition:* The magnitude of the update is proportional to the error $(y^{(i)} - h_\theta(x^{(i)}))$. If the prediction is accurate, parameters change very little; if the error is large, a larger adjustment is made.

> **Diagram Note - Gradient Descent Trajectory**
> A typical plot of a quadratic function (like $J(\theta)$) features concentric ellipses representing contours of the cost function. The trajectory of gradient descent initializes at some point and traces a path orthogonal to the contours, successively stepping down to the global minimum. 

### Batch Gradient Descent
To apply this to a training set of more than one example, we sum the gradients across all examples:
$$ \text{Repeat until convergence \{} $$
$$ \theta_j := \theta_j + \alpha \sum_{i=1}^n \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}_j \quad \text{(for every } j) $$
$$ \} $$

In vectorized form:
$$\theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}$$

This is called **batch gradient descent** because it scans the *entire* training set at each step. Since the least-squares cost function $J$ is a convex quadratic function, it has exactly one global optimum, so gradient descent is guaranteed to converge to the global minimum (assuming $\alpha$ is not too large).

### Stochastic (Incremental) Gradient Descent
When $n$ is large, batch gradient descent is computationally expensive. An alternative is **stochastic gradient descent**:
$$ \text{Loop \{} $$
$$ \quad \text{for } i = 1 \text{ to } n, \{ $$
$$ \qquad \theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}_j \quad \text{(for every } j) $$
$$ \quad \} $$
$$ \} $$

Vectorized update:
$$\theta := \theta + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}$$

Stochastic gradient descent updates the parameters upon seeing *each individual training example*. It often gets close to the minimum much faster than batch gradient descent, though it may oscillate around the minimum rather than strictly converging (unless $\alpha$ is slowly decayed to zero). It is generally preferred for large datasets.

---

## 1.2 The Normal Equations

Instead of an iterative algorithm, we can minimize $J(\theta)$ explicitly by finding its derivatives with respect to the $\theta_j$'s and setting them to zero.

### 1.2.1 Matrix Derivatives
For a function $f: \mathbb{R}^{n \times d} \mapsto \mathbb{R}$, we define the derivative of $f$ with respect to a matrix $A$ as:
$$ \nabla_A f(A) = \begin{bmatrix} \frac{\partial f}{\partial A_{11}} & \dots & \frac{\partial f}{\partial A_{1d}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial A_{n1}} & \dots & \frac{\partial f}{\partial A_{nd}} \end{bmatrix} $$

**Example:** If $A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}$ and $f(A) = \frac{3}{2}A_{11} + 5A_{12}^2 + A_{21}A_{22}$, then:
$$ \nabla_A f(A) = \begin{bmatrix} \frac{3}{2} & 10A_{12} \\ A_{22} & A_{21} \end{bmatrix} $$

### 1.2.2 Least Squares Revisited
We define the **design matrix** $X$ (an $n \times (d+1)$ matrix) containing the input values in its rows:
$$ X = \begin{bmatrix} - (x^{(1)})^T - \\ - (x^{(2)})^T - \\ \vdots \\ - (x^{(n)})^T - \end{bmatrix} $$

Let $\vec{y}$ be the $n$-dimensional vector containing all target values:
$$ \vec{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)} \end{bmatrix} $$

Since $h_\theta(x^{(i)}) = (x^{(i)})^T \theta$, we can write:
$$ X\theta - \vec{y} = \begin{bmatrix} (x^{(1)})^T \theta \\ \vdots \\ (x^{(n)})^T \theta \end{bmatrix} - \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(n)} \end{bmatrix} = \begin{bmatrix} h_\theta(x^{(1)}) - y^{(1)} \\ \vdots \\ h_\theta(x^{(n)}) - y^{(n)} \end{bmatrix} $$

Using the identity $z^T z = \sum_i z_i^2$, we can rewrite the cost function in matrix notation:
$$ J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) $$

To minimize $J$, we compute its gradient with respect to $\theta$. Using properties $\nabla_x b^T x = b$ and $\nabla_x x^T A x = 2Ax$ (for symmetric $A$), and that $a^T b = b^T a$:
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) \\
&= \frac{1}{2} \nabla_\theta \left( (X\theta)^T (X\theta) - (X\theta)^T \vec{y} - \vec{y}^T (X\theta) + \vec{y}^T \vec{y} \right) \\
&= \frac{1}{2} \nabla_\theta \left( \theta^T (X^T X) \theta - \vec{y}^T (X\theta) - \vec{y}^T (X\theta) \right) \\
&= \frac{1}{2} \nabla_\theta \left( \theta^T (X^T X) \theta - 2 (X^T \vec{y})^T \theta \right) \\
&= \frac{1}{2} \left( 2 X^T X \theta - 2 X^T \vec{y} \right) \\
&= X^T X \theta - X^T \vec{y}
\end{aligned}
$$

Setting this derivative to zero yields the **normal equations**:
$$ X^T X \theta = X^T \vec{y} $$

The closed-form value of $\theta$ that minimizes $J(\theta)$ is:
$$ \theta = (X^T X)^{-1} X^T \vec{y} $$
*(Note: This assumes $X^T X$ is invertible, which requires the features to be linearly independent and the number of examples $n$ to be greater than or equal to the number of features $d$.)*

---

## 1.3 Probabilistic Interpretation

Why is the least-squares cost function $J$ a reasonable choice? It arises naturally under certain probabilistic assumptions.

Assume the target variables and inputs are related by:
$$ y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)} $$
where $\epsilon^{(i)}$ is an error term capturing unmodeled effects or random noise. Assume the $\epsilon^{(i)}$ are independently and identically distributed (IID) according to a Gaussian distribution with mean zero and variance $\sigma^2$: $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$.

The density of $\epsilon^{(i)}$ is:
$$ p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( - \frac{(\epsilon^{(i)})^2}{2\sigma^2} \right) $$

This implies the distribution of $y^{(i)}$ given $x^{(i)}$ is parameterized by $\theta$:
$$ p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) $$
or $y^{(i)} | x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$.

The **likelihood function**, which represents the probability of the data given $\theta$, is:
$$ L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} | X; \theta) $$
By the independence assumption, this is the product of individual probabilities:
$$ L(\theta) = \prod_{i=1}^n p(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) $$

To determine the best parameters, we use the principle of **Maximum Likelihood Estimation (MLE)**, choosing $\theta$ to maximize $L(\theta)$. For simplicity, we maximize the strictly increasing **log-likelihood**, $\ell(\theta)$:
$$
\begin{aligned}
\ell(\theta) &= \log L(\theta) \\
&= \log \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) \\
&= \sum_{i=1}^n \log \left( \frac{1}{\sqrt{2\pi}\sigma} \exp\left( - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) \right) \\
&= n \log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{\sigma^2} \frac{1}{2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2
\end{aligned}
$$

Maximizing $\ell(\theta)$ is equivalent to minimizing the term:
$$ \frac{1}{2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2 $$
which is exactly $J(\theta)$, the ordinary least-squares cost function! 

*Conclusion:* Least-squares regression corresponds to finding the maximum likelihood estimate of $\theta$ under the assumption of Gaussian IID noise. (Note: the final choice of $\theta$ doesn't depend on $\sigma^2$).

---

## 1.4 Locally Weighted Linear Regression (LWR)

Feature selection is critical. Consider predicting $y$ from $x \in \mathbb{R}$:
> **Diagram Note - Model Fitting Examples**
> - **Underfitting (Left):** Fitting a linear $y = \theta_0 + \theta_1 x$ to curved data. The data clearly shows structure not captured by the simple model.
> - **Good Fit (Middle):** Adding an $x^2$ feature to fit a quadratic curve $y = \theta_0 + \theta_1 x + \theta_2 x^2$ captures the trend well.
> - **Overfitting (Right):** Fitting a 5th-order polynomial $y = \sum_{j=0}^5 \theta_j x^j$. The curve passes perfectly through the data points, but is highly erratic and would generalize poorly to new data.

**Locally Weighted Linear Regression (LWR)** makes feature selection less critical, given sufficient data.

In **standard linear regression**, to predict at a query point $x$:
1. Fit $\theta$ to minimize $\sum_i (y^{(i)} - \theta^T x^{(i)})^2$
2. Output $\theta^T x$

In **locally weighted linear regression (LWR)**, to predict at $x$:
1. Fit $\theta$ to minimize $\sum_i w^{(i)} (y^{(i)} - \theta^T x^{(i)})^2$
2. Output $\theta^T x$

Here, $w^{(i)}$ are non-negative weights that give higher influence to training examples that are closer to the query point $x$. A standard choice is:
$$ w^{(i)} = \exp\left( - \frac{(x^{(i)} - x)^2}{2\tau^2} \right) $$
*(For vector-valued $x$, this generalises to $w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^T(x^{(i)} - x)}{2\tau^2} \right)$ or $w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^T \Sigma^{-1} (x^{(i)} - x)}{2\tau^2} \right)$).*

- If $|x^{(i)} - x|$ is small, $w^{(i)} \approx 1$.
- If $|x^{(i)} - x|$ is large, $w^{(i)} \approx 0$.
- $\tau$ is the **bandwidth parameter** which controls how quickly the weight falls off with distance.
- *Note:* The formula resembles a Gaussian density, but $w^{(i)}$ are not probabilities or random variables.

### Parametric vs. Non-parametric Algorithms
- **Parametric:** Like standard linear regression. It has a fixed, finite number of parameters ($\theta_i$) fit to the data. The training set can be discarded after fitting.
- **Non-parametric:** Like LWR. You must keep the entire training set to make future predictions. The amount of data kept to represent the hypothesis $h$ grows linearly with the size of the training set.