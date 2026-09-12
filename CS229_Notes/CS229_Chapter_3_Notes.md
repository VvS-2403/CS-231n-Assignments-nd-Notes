---
title: CS229 Chapter 3 - Generalized Linear Models
tags: [cs229, machine-learning, glm, exponential-family, linear-regression, logistic-regression]
---

# Chapter 3: Generalized Linear Models (GLMs)

## Introduction

So far, we’ve seen two primary examples of predictive modeling:
1. **Regression:** $y|x; \theta \sim \mathcal{N}(\mu, \sigma^2)$
2. **Classification:** $y|x; \theta \sim \text{Bernoulli}(\phi)$

In both cases, $\mu$ and $\phi$ are functions of the inputs $x$ and parameters $\theta$. This chapter demonstrates that both ordinary least squares and logistic regression are special cases of a broader, more powerful family of models known as **Generalized Linear Models (GLMs)**. 

> [!NOTE] 
> The presentation of this material takes inspiration from Michael I. Jordan's *Learning in Graphical Models* and McCullagh and Nelder's *Generalized Linear Models*.

---

## 3.1 The Exponential Family

To understand GLMs, we first define **exponential family distributions**. A class of distributions belongs to the exponential family if its probability density (or mass) function can be written in the following form:

$$
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta)) \tag{3.1}
$$

### Terminology
- **$\eta$ (Natural/Canonical Parameter):** The parameter that determines the specific distribution within the family.
- **$T(y)$ (Sufficient Statistic):** For the distributions we consider, it is often simply the data itself, $T(y) = y$.
- **$a(\eta)$ (Log Partition Function):** The quantity $e^{-a(\eta)}$ acts as a normalization constant, ensuring that the distribution $p(y; \eta)$ sums or integrates to 1 over all possible values of $y$.
- **$b(y)$:** A base measure function that depends only on $y$, not on $\eta$.

A fixed choice of $T, a,$ and $b$ defines a specific *family* (or set) of distributions. By varying the natural parameter $\eta$, we get different distributions *within* this family.

### Example 1: The Bernoulli Distribution

The Bernoulli distribution models a binary outcome $y \in \{0, 1\}$ with mean $\phi$, denoted as $\text{Bernoulli}(\phi)$. 
- $p(y = 1; \phi) = \phi$
- $p(y = 0; \phi) = 1 - \phi$

We can express the Bernoulli distribution in the exponential family form as follows:

$$
\begin{aligned}
p(y; \phi) &= \phi^y (1 - \phi)^{1-y} \\
&= \exp\left(y \log \phi + (1 - y) \log(1 - \phi)\right) \\
&= \exp\left(\log\left(\frac{\phi}{1 - \phi}\right) y + \log(1 - \phi)\right)
\end{aligned}
$$

By matching this with Equation 3.1, we identify the components:
- **Natural Parameter:** $\eta = \log\left(\frac{\phi}{1 - \phi}\right)$
- **Sufficient Statistic:** $T(y) = y$
- **Log Partition Function:** $a(\eta) = -\log(1 - \phi) = \log(1 + e^\eta)$
- **Base Measure:** $b(y) = 1$

> [!TIP] Deriving the Sigmoid Function
> If we solve for $\phi$ in terms of $\eta$ from $\eta = \log\left(\frac{\phi}{1 - \phi}\right)$, we obtain:
> $$ \phi = \frac{1}{1 + e^{-\eta}} $$
> This is exactly the **sigmoid (logistic) function**! This relationship fundamentally explains why the sigmoid function is used in logistic regression.

### Example 2: The Gaussian Distribution

Consider the Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$. In linear regression, the value of $\sigma^2$ does not affect the optimal choice of $\theta$ and $h_\theta(x)$. To simplify the derivation, we can set $\sigma^2 = 1$. 

> [!NOTE] 
> If we leave $\sigma^2$ as a variable, the Gaussian distribution still belongs to the exponential family. $\eta \in \mathbb{R}^2$ becomes a 2-dimensional vector depending on both $\mu$ and $\sigma$. Alternatively, for GLMs, $\sigma^2$ can be treated using a more general definition: $p(y; \eta, \tau) = b(a, \tau) \exp\left(\frac{\eta^T T(y) - a(\eta)}{c(\tau)}\right)$, where $\tau$ is the dispersion parameter and $c(\tau) = \sigma^2$. For our purposes, setting $\sigma^2 = 1$ is sufficient.

With $\sigma^2 = 1$, the Gaussian density is:

$$
\begin{aligned}
p(y; \mu) &= \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2}(y - \mu)^2 \right) \\
&= \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2}y^2 \right) \cdot \exp\left( \mu y - \frac{1}{2}\mu^2 \right)
\end{aligned}
$$

Matching this with the exponential family form:
- **Natural Parameter:** $\eta = \mu$
- **Sufficient Statistic:** $T(y) = y$
- **Log Partition Function:** $a(\eta) = \frac{\mu^2}{2} = \frac{\eta^2}{2}$
- **Base Measure:** $b(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right)$

### Other Exponential Family Distributions
Many other distributions are members of the exponential family:
- **Multinomial:** For multi-class categorical data.
- **Poisson:** For modeling count data (e.g., estimating number of customers).
- **Gamma & Exponential:** For modeling continuous, non-negative random variables (e.g., time intervals).
- **Beta & Dirichlet:** For distributions over probabilities.

---

## 3.2 Constructing GLMs

Suppose you would like to build a model to estimate the number $y$ of customers arriving in your store (or page-views on your website) in any given hour, based on certain features $x$ (like store promotions, recent advertising, weather, day-of-week). We know the Poisson distribution usually gives a good model for numbers of visitors. Since Poisson is in the exponential family, we can easily apply a Generalized Linear Model (GLM).

### The Three GLM Assumptions (Design Choices)

To derive a GLM for predicting a random variable $y$ as a function of $x$, we make the following three assumptions about the conditional distribution of $y$ given $x$:

1. **Distribution of $y$:** Given $x$ and $\theta$, the conditional distribution of $y$ belongs to the exponential family with natural parameter $\eta$:
   $$ y | x; \theta \sim \text{ExponentialFamily}(\eta) $$
2. **Prediction Goal:** Our goal is to predict the expected value of $T(y)$ given $x$. Since typically $T(y) = y$, our hypothesis function $h(x)$ must satisfy:
   $$ h_\theta(x) = E[y|x; \theta] $$
   *(Note: This is satisfied in both linear regression and logistic regression.)*
3. **Linearity of Natural Parameter:** The natural parameter $\eta$ and the inputs $x$ are related linearly:
   $$ \eta = \theta^T x $$
   *(Or if $\eta$ is vector-valued, $\eta_i = \theta_i^T x$.)*

These assumptions result in an elegant class of learning algorithms that are effective for modeling different distributions over $y$.

---

### 3.2.1 Ordinary Least Squares as a GLM

Ordinary least squares regression perfectly fits the GLM framework for a continuous response variable $y$:
- **Assumption 1:** We model $y | x; \theta \sim \mathcal{N}(\mu, \sigma^2)$. As shown earlier, for a Gaussian, $\mu = \eta$.
- **Assumption 2:** Our hypothesis function predicts the expected value: $h_\theta(x) = E[y|x; \theta] = \mu$.
- **Assumption 3:** We assume a linear relationship: $\eta = \theta^T x$.

Putting it all together:
$$
h_\theta(x) = E[y|x; \theta] = \mu = \eta = \theta^T x
$$
This neatly derives the standard linear regression hypothesis!

---

### 3.2.2 Logistic Regression as a GLM

For binary classification, the target variable is $y \in \{0, 1\}$. 
- **Assumption 1:** We model $y|x; \theta$ using a Bernoulli distribution. From our Bernoulli derivation, $\phi = 1/(1 + e^{-\eta})$.
- **Assumption 2:** The expected value of a Bernoulli distribution is its mean parameter $\phi$, so $h_\theta(x) = E[y|x; \theta] = \phi$.
- **Assumption 3:** The natural parameter is modeled linearly: $\eta = \theta^T x$.

Putting it all together:
$$
h_\theta(x) = E[y|x; \theta] = \phi = \frac{1}{1 + e^{-\eta}} = \frac{1}{1 + e^{-\theta^T x}}
$$
This beautifully explains the origin of the logistic hypothesis function $h_\theta(x) = 1/(1 + e^{-\theta^T x})$! Once we assume $y$ conditioned on $x$ is Bernoulli, the logistic function arises as a direct consequence of the GLM definition.

---

## 3.3 Terminology: Canonical Functions

- **Canonical Response Function $g(\eta)$:** The function giving the distribution's mean as a function of the natural parameter.
  $$ g(\eta) = E[T(y); \eta] $$
- **Canonical Link Function $g^{-1}$:** The inverse of the response function. 

**Examples:**
- For the **Gaussian** family, the canonical response function is the **identity function**.
- For the **Bernoulli** family, the canonical response function is the **logistic (sigmoid) function**.

> [!NOTE] 
> *Note on Notation:* Many texts use $g$ to denote the link function, and $g^{-1}$ for the response function. However, the notation used here is inherited from early machine learning literature and will be consistent throughout the rest of the CS229 material.