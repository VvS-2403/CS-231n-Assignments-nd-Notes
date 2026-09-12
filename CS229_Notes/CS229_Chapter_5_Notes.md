# Chapter 5: Kernel Methods

## 5.1 Feature Maps
In linear regression, we typically fit a linear function of the input $x$ (e.g., predicting housing price $y$ from living area $x$). However, when $y$ is a non-linear function of $x$, we need a more expressive family of models. 

We can map the original input into a different set of variables using a **feature map**. For instance, to fit a cubic function $y = \theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0$, we define a function $\phi : \mathbb{R} \to \mathbb{R}^4$:

$$ \phi(x) = \begin{bmatrix} 1 \\ x \\ x^2 \\ x^3 \end{bmatrix} \in \mathbb{R}^4 $$

Let $\theta \in \mathbb{R}^4$ be the vector containing $\theta_0, \theta_1, \theta_2, \theta_3$. The cubic function can then be rewritten as a linear function of the new features:
$$ \theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0 = \theta^T \phi(x) $$

**Terminology:**
- **Input Attributes:** The "original" input value of a problem (e.g., $x$, the living area).
- **Feature Variables:** The new set of quantities $\phi(x)$ mapped from the original input.
- **Feature Map ($\phi$):** The function that maps the attributes to the features.

## 5.2 LMS (Least Mean Squares) with Features
We want to derive the gradient descent algorithm for fitting the model $\theta^T \phi(x)$. 

For the ordinary least squares problem (fitting $\theta^T x$), the batch gradient descent update is:
$$ \theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)} $$
$$ \theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T x^{(i)} \right) x^{(i)} $$

Let $\phi : \mathbb{R}^d \to \mathbb{R}^p$ be a feature map. To fit $\theta^T \phi(x)$ (where $\theta \in \mathbb{R}^p$), we replace $x^{(i)}$ with $\phi(x^{(i)})$ in the update rule.

**Batch Gradient Descent Update:**
$$ \theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) $$

**Stochastic Gradient Descent (SGD) Update:**
$$ \theta := \theta + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) $$

## 5.3 LMS with the Kernel Trick
The updates become computationally expensive when $\phi(x)$ is high-dimensional. For example, if we extend the feature map to include all monomials of $x \in \mathbb{R}^d$ up to degree 3, the dimension of $\phi(x)$ becomes $p = O(d^3)$. For $d=1000$, $p \approx 10^9$, making each update prohibitively slow and memory-intensive ($10^6$ times slower than ordinary least squares).

The **kernel trick** allows us to perform these updates without explicitly storing or computing the high-dimensional vector $\theta$.

### Implicit Representation of $\theta$
Assume we initialize $\theta = 0$. We observe that $\theta$ can always be represented as a linear combination of the feature vectors $\phi(x^{(1)}), \ldots, \phi(x^{(n)})$.

At initialization, $\theta = \sum_{i=1}^n 0 \cdot \phi(x^{(i)})$. Assuming at some step $\theta$ is represented as:
$$ \theta = \sum_{i=1}^n \beta_i \phi(x^{(i)}) $$
for some coefficients $\beta_1, \ldots, \beta_n \in \mathbb{R}$, then the next update is:
$$ \theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) $$
$$ = \sum_{i=1}^n \beta_i \phi(x^{(i)}) + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) $$
$$ = \sum_{i=1}^n \underbrace{\left( \beta_i + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \right)}_{\text{new } \beta_i} \phi(x^{(i)}) $$

We can update the coefficients $\beta_i$ instead of $\theta$:
$$ \beta_i := \beta_i + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) $$

Replacing $\theta$ with $\sum_{j=1}^n \beta_j \phi(x^{(j)})$, we get:
$$ \forall i \in \{1, \ldots, n\}, \quad \beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^n \beta_j \phi(x^{(j)})^T \phi(x^{(i)}) \right) $$

Here, we rewrite $\phi(x^{(j)})^T \phi(x^{(i)})$ as the inner product $\langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle$. 

### The Kernel Function
Computing $\langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle$ directly would still take $O(p)$ time. However, two properties help us:
1. We can pre-compute pairwise inner products $\langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle$.
2. Computing the inner product can be very efficient without explicitly computing $\phi(x)$. 

For the degree-3 monomial feature map, the inner product is:
$$ \langle \phi(x), \phi(z) \rangle = 1 + \sum_{i=1}^d x_i z_i + \sum_{i,j \in \{1, \dots, d\}} x_i x_j z_i z_j + \sum_{i,j,k \in \{1, \dots, d\}} x_i x_j x_k z_i z_j z_k $$
$$ = 1 + \sum_{i=1}^d x_i z_i + \left( \sum_{i=1}^d x_i z_i \right)^2 + \left( \sum_{i=1}^d x_i z_i \right)^3 $$
$$ = 1 + \langle x, z \rangle + \langle x, z \rangle^2 + \langle x, z \rangle^3 $$
Thus, we can compute $\langle \phi(x), \phi(z) \rangle$ in $O(d)$ time instead of $O(d^3)$.

We define the **Kernel** corresponding to the feature map $\phi$ as a function $K : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$:
$$ K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle $$

### The Final Algorithm
1. Pre-compute $K(x^{(i)}, x^{(j)}) = \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle$ for all $i, j \in \{1, \dots, n\}$. Initialize $\beta := 0$.
2. Loop:
$$ \forall i \in \{1, \dots, n\}, \quad \beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^n \beta_j K(x^{(i)}, x^{(j)}) \right) $$

In vector notation (where $K$ is the $n \times n$ matrix with $K_{ij} = K(x^{(i)}, x^{(j)})$ and $\vec{y}$ is the target vector):
$$ \beta := \beta + \alpha (\vec{y} - K\beta) $$

To make a prediction for a new test example $x$, we only need the representation $\beta$:
$$ \theta^T \phi(x) = \sum_{i=1}^n \beta_i \phi(x^{(i)})^T \phi(x) = \sum_{i=1}^n \beta_i K(x^{(i)}, x) $$
Fundamentally, everything we need to know about the feature map $\phi(\cdot)$ is encapsulated within the kernel function $K(\cdot, \cdot)$.

## 5.4 Properties of Kernels
Since the algorithm doesn't explicitly access $\phi$, we can work backward: we can select a Kernel function $K(x, z)$ directly, verifying that there exists *some* $\phi$ such that $K(x, z) = \phi(x)^T \phi(z)$. This changes the interface from selecting feature maps to selecting kernel functions.

### Examples of Kernels
**1. Quadratic Kernel:**
Consider $K(x, z) = (x^T z)^2$.
$$ K(x, z) = \left( \sum_{i=1}^d x_i z_i \right) \left( \sum_{j=1}^d x_j z_j \right) = \sum_{i=1}^d \sum_{j=1}^d x_i x_j z_i z_j = \sum_{i,j=1}^d (x_i x_j)(z_i z_j) $$
This corresponds to a feature map $\phi(x)$ containing all second-order terms $x_i x_j$. Computation is $O(d)$, whereas calculating $\phi(x)$ explicitly would take $O(d^2)$.

**2. Polynomial Kernel:**
$$ K(x, z) = (x^T z + c)^2 $$
$$ = \sum_{i,j=1}^d (x_i x_j)(z_i z_j) + \sum_{i=1}^d (\sqrt{2c} x_i)(\sqrt{2c} z_i) + c^2 $$
This maps to a feature space containing first and second-order terms, controlled by weight $c$.
More broadly, $K(x, z) = (x^T z + c)^k$ corresponds to all monomials up to degree $k$ in an $\binom{d+k}{k}$-dimensional feature space, yet always takes only $O(d)$ time to compute.

**3. Gaussian Kernel (Kernels as Similarity Metrics):**
Kernels intuitively act as similarity metrics. If $\phi(x)$ and $\phi(z)$ are close, $K(x, z)$ is large; if they are nearly orthogonal, it is small.
$$ K(x, z) = \exp \left( - \frac{||x - z||_2^2}{2\sigma^2} \right) $$
This is a valid kernel corresponding to an **infinite-dimensional** feature mapping. It is close to $1$ when $x$ and $z$ are similar, and near $0$ when they are far apart.

### Necessary and Sufficient Conditions for Valid Kernels
Suppose $K$ is a valid kernel corresponding to some feature mapping $\phi$. For any finite set of $n$ points $\{x^{(1)}, \ldots, x^{(n)}\}$, let the **kernel matrix** $K \in \mathbb{R}^{n \times n}$ be defined as $K_{ij} = K(x^{(i)}, x^{(j)})$.

**Necessary Condition:**
- **Symmetry:** $K_{ij} = \phi(x^{(i)})^T \phi(x^{(j)}) = \phi(x^{(j)})^T \phi(x^{(i)}) = K_{ji}$. $K$ must be symmetric.
- **Positive Semi-definiteness:** Let $\phi_k(x)$ denote the $k$-th coordinate of $\phi(x)$. For any vector $z$:
$$ z^T K z = \sum_i \sum_j z_i K_{ij} z_j = \sum_i \sum_j z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j $$
$$ = \sum_i \sum_j z_i \sum_k \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j = \sum_k \sum_i \sum_j z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j $$
$$ = \sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right)^2 \geq 0 $$
Since $z$ is arbitrary, $K$ is positive semi-definite ($K \ge 0$).

**Sufficient Condition (Mercer's Theorem):**
> **Theorem (Mercer):** Let $K : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ be given. Then for $K$ to be a valid (Mercer) kernel, it is necessary and sufficient that for any $\{x^{(1)}, \ldots, x^{(n)}\}$ ($n < \infty$), the corresponding kernel matrix is symmetric positive semi-definite.

### Applications of Kernel Methods
- **Digit Recognition:** SVMs with polynomial or Gaussian kernels achieve excellent performance taking raw pixel intensities as 256-dimensional inputs, completely without prior vision knowledge about pixel adjacency.
- **String Matching:** For non-vector inputs like strings of amino acids (proteins), a feature vector $\phi(x)$ could count occurrences of all length-$k$ substrings. While $\phi(x)$ for English letters has a huge dimension of $26^k$, $K(x, z) = \phi(x)^T \phi(z)$ can be computed efficiently using dynamic programming string matching algorithms without explicitly building $\phi$.
- **The Kernel Trick:** The kernel trick extends far beyond linear regression and SVMs. If *any* learning algorithm can be rewritten in terms of only inner products $\langle x, z \rangle$ between input attribute vectors, replacing it with $K(x, z)$ allows the algorithm to magically and efficiently operate in the high-dimensional feature space corresponding to $K$ (e.g., deriving the kernel perceptron algorithm).