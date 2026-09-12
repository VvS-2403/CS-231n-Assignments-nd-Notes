# CS229 Chapter 13: Independent Components Analysis (ICA)

## Introduction
Our next topic is **Independent Components Analysis (ICA)**. Similar to PCA, this will find a new basis in which to represent our data. However, the goal is very different.

### Motivating Example: The Cocktail Party Problem
Consider the "cocktail party problem." Here, $d$ speakers are speaking simultaneously at a party, and any microphone placed in the room records only an overlapping combination of the $d$ speakers' voices. But let's say we have $d$ different microphones placed in the room, and because each microphone is a different distance from each of the speakers, it records a different combination of the speakers' voices. Using these microphone recordings, can we separate out the original $d$ speakers' speech signals?

### Formalizing the Problem
To formalize this problem, we imagine that there is some data $s \in \mathbb{R}^d$ that is generated via $d$ independent sources. What we observe is
$$x = As$$
where $A$ is an unknown square matrix called the **mixing matrix**. Repeated observations give us a dataset $\{x^{(i)}; i = 1, \dots, n\}$, and our goal is to recover the sources $s^{(i)}$ that had generated our data ($x^{(i)} = As^{(i)}$).

In our cocktail party problem:
- $s^{(i)}$ is a $d$-dimensional vector, and $s^{(i)}_j$ is the sound that speaker $j$ was uttering at time $i$. 
- $x^{(i)}$ is a $d$-dimensional vector, and $x^{(i)}_j$ is the acoustic reading recorded by microphone $j$ at time $i$.

Let $W = A^{-1}$ be the **unmixing matrix**. Our goal is to find $W$, so that given our microphone recordings $x^{(i)}$, we can recover the sources by computing:
$$s^{(i)} = Wx^{(i)}$$

For notational convenience, we let $w_i^T$ denote the $i$-th row of $W$, so that:
$$W = \begin{bmatrix} \text{---} & w_1^T & \text{---} \\ & \vdots & \\ \text{---} & w_d^T & \text{---} \end{bmatrix}$$

Thus, $w_i \in \mathbb{R}^d$, and the $j$-th source can be recovered as:
$$s^{(i)}_j = w_j^T x^{(i)}$$

## 13.1 ICA Ambiguities

To what degree can $W = A^{-1}$ be recovered? If we have no prior knowledge about the sources and the mixing matrix, there are some inherent ambiguities in $A$ that are impossible to recover given only the $x^{(i)}$'s.

### 1. Permutation Ambiguity
Let $P$ be any $d \times d$ permutation matrix (a matrix where each row and column has exactly one "1"). Examples of permutation matrices include:
$$P = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}; \quad P = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}; \quad P = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

If $z$ is a vector, then $Pz$ is a vector that contains a permuted version of $z$'s coordinates. Given only the $x^{(i)}$'s, there will be no way to distinguish between $W$ and $PW$. Specifically, **the permutation of the original sources is ambiguous**. Fortunately, this does not matter for most applications.

### 2. Scaling and Sign Ambiguity
There is no way to recover the correct scaling of the $w_i$'s. 
- If $A$ were replaced with $2A$, and every $s^{(i)}$ were replaced with $(0.5)s^{(i)}$, our observed $x^{(i)} = 2A \cdot (0.5)s^{(i)}$ would still be the same. 
- More broadly, if a single column of $A$ were scaled by a factor of $\alpha$, and the corresponding source were scaled by a factor of $1/\alpha$, there is no way to determine that this had happened given only the $x^{(i)}$'s. 
- Thus, **we cannot recover the "correct" scaling of the sources**. 

However, for applications like the cocktail party problem, this ambiguity also does not matter:
- Scaling a speaker's speech signal $s^{(i)}_j$ by a positive factor $\alpha$ affects only the *volume* of that speaker's speech. 
- Sign changes do not matter; $s^{(i)}_j$ and $-s^{(i)}_j$ sound identical when played on a speaker. 

Thus, if the $w_i$ found by an algorithm is scaled by any non-zero real number, the corresponding recovered source $s_i = w_i^T x$ will be scaled by the same factor, which usually does not matter (this also applies to ICA for brain/MEG data).

### The Problem with Gaussian Data
Are these the only sources of ambiguity in ICA? Yes, **so long as the sources $s_i$ are non-Gaussian**. 

To see what the difficulty is with Gaussian data, consider an example where $n = 2$ and $s \sim \mathcal{N}(0, I)$ (where $I$ is the $2 \times 2$ identity matrix). The contours of the density of the standard normal distribution $\mathcal{N}(0, I)$ are circles centered on the origin, and the density is rotationally symmetric.

Suppose we observe $x = As$, where $A$ is our mixing matrix. Then, the distribution of $x$ will be Gaussian, $x \sim \mathcal{N}(0, AA^T)$, since:
$$\mathbb{E}_{s \sim \mathcal{N}(0, I)}[x] = \mathbb{E}[As] = A\mathbb{E}[s] = 0$$
$$\text{Cov}[x] = \mathbb{E}_{s \sim \mathcal{N}(0, I)}[xx^T] = \mathbb{E}[Ass^TA^T] = A\mathbb{E}[ss^T]A^T = A \cdot \text{Cov}[s] \cdot A^T = AA^T$$

Now, let $R$ be an arbitrary orthogonal (rotation/reflection) matrix, so that $RR^T = R^TR = I$, and let $A' = AR$. If the data had been mixed according to $A'$ instead of $A$, we would have observed $x' = A's$. The distribution of $x'$ is also Gaussian, $x' \sim \mathcal{N}(0, AA^T)$, since:
$$\mathbb{E}_{s \sim \mathcal{N}(0, I)}[x'(x')^T] = \mathbb{E}[A'ss^T(A')^T] = \mathbb{E}[ARss^T(AR)^T] = ARR^TA^T = AA^T$$

Hence, whether the mixing matrix is $A$ or $A'$, we would observe data from an $\mathcal{N}(0, AA^T)$ distribution. **There is an arbitrary rotational component in the mixing matrix that cannot be determined from the data, and we cannot recover the original sources.** Thus, ICA cannot model Gaussian data.

> [!tip] Non-Gaussian Requirement
> The impossibility of recovering sources for Gaussian data stems from the fact that the multivariate standard normal distribution is rotationally symmetric. However, so long as the data is **not** Gaussian, it is possible (given enough data) to recover the $d$ independent sources.

## 13.2 Densities and Linear Transformations

Before deriving the ICA algorithm, let's discuss the effect of linear transformations on densities.

Suppose a random variable $s \in \mathbb{R}$ is drawn according to some density $p_s(s)$. Let the random variable $x \in \mathbb{R}$ be defined as $x = As$ for some $A \in \mathbb{R}$. What is the density $p_x(x)$?

Let $W = A^{-1}$. It is tempting to compute $s = Wx$, evaluate $p_s$ at that point, and conclude that $p_x(x) = p_s(Wx)$. **However, this is incorrect.**

> [!example] Counterexample for $p_x(x) = p_s(Wx)$
> Let $s \sim \text{Uniform}[0, 1]$, so $p_s(s) = 1\{0 \leq s \leq 1\}$. 
> Let $A = 2$, so $x = 2s$. Clearly, $x$ is distributed uniformly in the interval $[0, 2]$. 
> Thus, its density is $p_x(x) = (0.5)1\{0 \leq x \leq 2\}$. 
> This does *not* equal $p_s(Wx)$ where $W = 0.5 = A^{-1}$. 

Instead, the correct formula is:
$$p_x(x) = p_s(Wx)|W|$$

More generally, if $s$ is a vector-valued distribution with density $p_s$, and $x = As$ for a square, invertible matrix $A$, then the density of $x$ is given by:
$$p_x(x) = p_s(Wx) \cdot |W|$$
where $W = A^{-1}$ and $|W|$ is the determinant of $W$.

> [!info] Intuition from Linear Algebra
> Let $A \in \mathbb{R}^{d \times d}$, and let $W = A^{-1}$. Let $C_1 = [0, 1]^d$ be the $d$-dimensional hypercube, and define $C_2 = \{As : s \in C_1\} \subseteq \mathbb{R}^d$ as the image of $C_1$ under the mapping $A$. 
> The volume of $C_2$ is given by $|A|$. 
> If $s$ is uniformly distributed in $[0, 1]^d$, its density is $p_s(s) = 1\{s \in C_1\}$. Then $x$ will be uniformly distributed in $C_2$. 
> Its density is $p_x(x) = 1\{x \in C_2\} / \text{vol}(C_2)$ (since it must integrate to 1).
> Using $1 / \text{vol}(C_2) = 1 / |A| = |A^{-1}| = |W|$, we get:
> $$p_x(x) = 1\{x \in C_2\}|W| = 1\{Wx \in C_1\}|W| = p_s(Wx)|W|$$

## 13.3 ICA Algorithm

We can now derive the ICA algorithm (originally by Bell and Sejnowski) using **maximum likelihood estimation**.

We suppose that the distribution of each source $s_j$ is given by a density $p_s$, and that the joint distribution of the sources $s$ is given by:
$$p(s) = \prod_{j=1}^d p_s(s_j)$$
By modeling the joint distribution as a product of marginals, we capture the assumption that the **sources are independent**.

Using our formulas from the previous section, this implies the following density on $x = As = W^{-1}s$:
$$p(x) = \left( \prod_{j=1}^d p_s(w_j^T x) \right) \cdot |W|$$

### Choosing the Source Density $p_s$
All that remains is to specify a density for the individual sources $p_s$. 
Given a real-valued random variable $z$, its cumulative distribution function (CDF) $F$ is defined by $F(z_0) = P(z \leq z_0) = \int_{-\infty}^{z_0} p_z(z) dz$, and the density is the derivative of the CDF: $p_z(z) = F'(z)$.

To specify a density for the $s_i$'s, we specify a CDF for it (a monotonic function that increases from zero to one). We cannot choose the Gaussian CDF since ICA doesn't work on Gaussian data. Instead, we choose a reasonable "default" CDF that slowly increases from $0$ to $1$: the **sigmoid function**:
$$g(s) = \frac{1}{1 + e^{-s}}$$
Hence, our density is the derivative of the sigmoid function:
$$p_s(s) = g'(s)$$

> [!note] Data Preprocessing
> The sigmoid function is a reasonable default that works well for many problems. If you have prior knowledge about the sources' densities, you should substitute that in. 
> Our assumption that $p_s(s) = g'(s)$ implies $\mathbb{E}[s] = 0$ (since the derivative of the logistic function is symmetric). This implies $\mathbb{E}[x] = \mathbb{E}[As] = 0$. Therefore, **the data $x^{(i)}$ must be preprocessed to have zero mean**, or it naturally has zero mean (like acoustic signals).

### Maximum Likelihood Estimation
The square matrix $W$ is the parameter in our model. Given a training set $\{x^{(i)}; i = 1, \dots, n\}$, the **log likelihood** is given by:
$$\ell(W) = \sum_{i=1}^n \left( \sum_{j=1}^d \log g'(w_j^T x^{(i)}) + \log |W| \right)$$

We would like to maximize this in terms of $W$. By taking derivatives and using the fact that $\nabla_W |W| = |W|(W^{-1})^T$, we can derive a **stochastic gradient ascent** learning rule. 

For a single training example $x^{(i)}$, the update rule is:
$$W := W + \alpha \left( \begin{bmatrix} 1 - 2g(w_1^T x^{(i)}) \\ 1 - 2g(w_2^T x^{(i)}) \\ \vdots \\ 1 - 2g(w_d^T x^{(i)}) \end{bmatrix} {x^{(i)}}^T + (W^T)^{-1} \right)$$
where $\alpha$ is the learning rate.

After the algorithm converges, we then compute:
$$s^{(i)} = Wx^{(i)}$$
to recover the original sources.

> [!tip] Accelerating Convergence
> When writing down the likelihood, we implicitly assumed that the $x^{(i)}$'s were independent of each other, meaning the likelihood of the training set was $\prod_i p(x^{(i)}; W)$. 
> This assumption is clearly incorrect for speech data and other time series where successive $x^{(i)}$'s are correlated. However, having correlated training examples will not hurt the performance of the algorithm if we have sufficient data. 
> For stochastic gradient ascent, it sometimes **helps accelerate convergence if we visit training examples in a randomly permuted order** (i.e., run stochastic gradient ascent on a randomly shuffled copy of the training set).