# Chapter 12: Principal Components Analysis (PCA)

## 1. Introduction
In this chapter, we develop a method called **Principal Components Analysis (PCA)**. The goal of PCA is to identify the subspace in which the data approximately lies.
- **Computational Efficiency:** PCA is highly computationally efficient. It requires only an eigenvector calculation, which can be easily done using functions like `eig` in Matlab.

### Motivating Examples
#### 1. Automobile Attributes
Suppose we have a dataset $\{x^{(i)}; i = 1, \dots, n\}$ of attributes of $n$ different types of automobiles (e.g., maximum speed, turn radius). Let $x^{(i)} \in \mathbb{R}^d$ for each $i$ (where $d \ll n$). 
- Unknown to us, two different attributes (say $x_i$ and $x_j$) might give a car's maximum speed measured in miles per hour (mph) and kilometers per hour (kph) respectively.
- These two attributes are therefore almost linearly dependent, up to small differences introduced by rounding off.
- The data really lies approximately on an $n - 1$ dimensional subspace. PCA can automatically detect and perhaps remove this redundancy.

#### 2. RC Helicopter Pilots
Consider a dataset resulting from a survey of pilots for radio-controlled (RC) helicopters.
- Let $x_1^{(i)}$ measure the piloting skill of pilot $i$.
- Let $x_2^{(i)}$ capture how much they enjoy flying.
Because RC helicopters are extremely difficult to fly, only the most committed students who truly enjoy flying become good pilots.
- The two attributes $x_1$ and $x_2$ are strongly correlated.
- We might posit that the data actually lies along some diagonal axis (the $u_1$ direction) capturing the intrinsic piloting "karma" of a person, with only a small amount of noise lying off this axis. 
- *Diagram Description:* A scatter plot with the $x_1$-axis as "skill" and $x_2$-axis as "enjoyment" shows data points scattered along a diagonal line defined by a vector $u_1$. The goal is to automatically compute this $u_1$ direction.

## 2. Data Preprocessing
Prior to running PCA per se, it is typical to first preprocess the data to ensure that each feature has a mean of 0 and a variance of 1.

### Normalization Step
Subtract the mean and divide by the empirical standard deviation for each feature:
$$
x_j^{(i)} \leftarrow \frac{x_j^{(i)} - \mu_j}{\sigma_j}
$$
Where:
- **Mean:** $\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_j^{(i)}$
- **Variance:** $\sigma_j^2 = \frac{1}{n} \sum_{i=1}^{n} \left( x_j^{(i)} - \mu_j \right)^2$

#### Properties of Normalization:
- **Zero-mean ($\mu_j$ subtraction):** Zeros out the mean. Can be omitted if the data is already known to have zero mean (e.g., time series corresponding to speech or other acoustic signals).
- **Unit variance ($\sigma_j$ division):** Rescales each coordinate to have unit variance. This ensures different attributes are treated on the same scale (e.g., if $x_1$ was a car's top speed in mph taking values in the 100s, vs. $x_2$ as the number of seats taking values 2-4).
- **When to omit rescaling:** Rescaling may be omitted if there's *a priori* knowledge that all attributes are all on the same scale (e.g., in a grayscale image dataset, where each $x_j^{(i)} \in \{0, 1, \dots, 255\}$ corresponds to the pixel intensity).

## 3. Formulating the PCA Problem
After normalizing our data, the goal is to compute the "major axis of variation," denoted as the unit vector $u$ — that is, the direction on which the data approximately lies.

### Maximizing Variance of Projected Data
One way to pose this problem is to find the unit vector $u$ such that when the data is projected onto the direction corresponding to $u$, the variance of the projected data is maximized. 
- Intuitively, the data starts off with some amount of variance/information in it. 
- We want to choose a direction $u$ so that if we approximate the data as lying in that direction/subspace, as much of this variance as possible is retained.

#### Diagram Descriptions
1. **High Variance Projection (Preferred):** A scatter plot of data points with a line passing through them representing direction $u$. The circles denoting projections of the original data onto this line show a fairly large variance, with points tending to be far from zero.
2. **Low Variance Projection (Suboptimal):** The same data with an alternative perpendicular line choice for direction $u$. The projections (circles) are much closer to the origin, indicating a significantly smaller variance. 

PCA aims to automatically select the direction corresponding to the first figure (maximizing variance).

## 4. Mathematical Derivation of PCA
Given a unit vector $u$ and a point $x$, the length of the projection of $x$ onto $u$ is given by $x^T u$. 
- If $x^{(i)}$ is a point in our dataset, its projection onto $u$ is distance $(x^{(i)})^T u$ from the origin.

To maximize the variance of the projections, we want to choose a unit-length vector $u$ that maximizes:
$$
\frac{1}{n} \sum_{i=1}^{n} \left( (x^{(i)})^T u \right)^2 = \frac{1}{n} \sum_{i=1}^{n} u^T x^{(i)} (x^{(i)})^T u 
$$
$$
= u^T \left( \frac{1}{n} \sum_{i=1}^{n} x^{(i)} (x^{(i)})^T \right) u
$$

### The Covariance Matrix
We easily recognize that the matrix term in the middle is the empirical covariance matrix of the data (assuming zero mean):
$$
\Sigma = \frac{1}{n} \sum_{i=1}^{n} x^{(i)} (x^{(i)})^T
$$

### Finding the Principal Components
Maximizing $u^T \Sigma u$ subject to the constraint $\|u\|_2 = 1$ gives the **principal eigenvector** of $\Sigma$.
> **Note (Footnote 1):** This can be derived using the method of Lagrange multipliers. We maximize $u^T \Sigma u$ subject to $u^T u = 1$. You should be able to show that $\Sigma u = \lambda u$, for some $\lambda$, which implies $u$ is an eigenvector of $\Sigma$, with eigenvalue $\lambda$.

To project the data into a $k$-dimensional subspace ($k < d$), we choose $u_1, \dots, u_k$ to be the top $k$ eigenvectors of $\Sigma$. 
- Because $\Sigma$ is symmetric, the $u_i$'s will (or always can be chosen to be) orthogonal to each other (Footnote 2).
- The vectors $u_1, \dots, u_k$ form a new, orthogonal basis for the data and are called the **first $k$ principal components** of the data.

## 5. Dimensionality Reduction
To represent a data point $x^{(i)}$ in this new $k$-dimensional basis, we compute the corresponding vector $y^{(i)}$:
$$
y^{(i)} = 
\begin{bmatrix}
u_1^T x^{(i)} \\
u_2^T x^{(i)} \\
\vdots \\
u_k^T x^{(i)}
\end{bmatrix} \in \mathbb{R}^k
$$
Whereas $x^{(i)} \in \mathbb{R}^d$, the vector $y^{(i)}$ now gives a lower, $k$-dimensional approximation/representation for $x^{(i)}$. PCA is therefore also referred to as a **dimensionality reduction** algorithm.

### Remark on Preserving Variability
By properties of eigenvectors, of all possible orthogonal bases $u_1, \dots, u_k$, the one that we have chosen maximizes $\sum_i \|y^{(i)}\|_2^2$. Thus, our choice of basis preserves as much variability as possible in the original data. 
- *Alternative perspective:* PCA can also be derived by picking the basis that minimizes the approximation error arising from projecting the data onto the $k$-dimensional subspace spanned by them.

## 6. Applications of PCA
PCA has many practical applications, including:
1. **Compression:** Representing high-dimensional $x^{(i)}$'s with lower-dimensional $y^{(i)}$'s.
2. **Visualization:** Reducing high-dimensional data to $k=2$ or $3$ dimensions allows us to plot $y^{(i)}$'s (e.g., plotting the automobiles dataset to see what cars are similar to each other and what groups of cars may cluster together).
3. **Preprocessing for Supervised Learning:** Reducing the input dimensionality before running supervised learning algorithms can:
   - Provide computational benefits.
   - Reduce the complexity of the hypothesis class, helping to avoid **overfitting** (e.g., linear classifiers over lower-dimensional inputs will have a smaller VC dimension).
4. **Noise Reduction:** As seen in the RC pilot example, PCA can isolate the intrinsic signal ("piloting karma") from noisy measures.

### Eigenfaces Method (Application Example)
- Applied to face images, where each $x^{(i)} \in \mathbb{R}^{10000}$ (each coordinate corresponding to a pixel intensity value in a $100 \times 100$ image).
- Using PCA, each image $x^{(i)}$ is represented with a much lower-dimensional $y^{(i)}$.
- **Goal:** The principal components found retain the interesting, systematic variations between faces (capturing what a person really looks like) while discarding the "noise" introduced by minor lighting variations, slightly different imaging conditions, and so on.
- **Face-Matching Algorithm:** We then measure distances between faces $i$ and $j$ by computing $\|y^{(i)} - y^{(j)}\|_2$ in the reduced dimension, resulting in a surprisingly good face-matching and retrieval algorithm.