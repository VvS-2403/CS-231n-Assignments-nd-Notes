# CS229 Notes: Chapter 4 - Generative Learning Algorithms

## Introduction
So far, learning algorithms (like logistic regression and the perceptron) have focused on modeling $p(y|x; \theta)$, the conditional distribution of $y$ given $x$, or mapping inputs directly to labels. These are called **discriminative learning algorithms**. Discriminative algorithms try to find a decision boundary (e.g., a straight line) that separates classes.

**Generative learning algorithms** take a different approach: they try to model $p(x|y)$ (and $p(y)$). For a classification problem (e.g., separating dogs where $y=0$ from elephants where $y=1$):
- $p(x|y=0)$ models the distribution of dogs' features.
- $p(x|y=1)$ models the distribution of elephants' features.
- $p(y)$ models the class priors.

Using Bayes rule, we can derive the posterior distribution on $y$ given $x$:
$$ p(y|x) = \frac{p(x|y)p(y)}{p(x)} $$

Where the denominator is given by the law of total probability:
$$ p(x) = p(x|y=1)p(y=1) + p(x|y=0)p(y=0) $$

For prediction, we do not actually need to calculate the denominator since it is independent of $y$:
$$ \arg\max_y p(y|x) = \arg\max_y \frac{p(x|y)p(y)}{p(x)} = \arg\max_y p(x|y)p(y) $$

---

## 4.1 Gaussian Discriminant Analysis (GDA)
Gaussian Discriminant Analysis is a generative algorithm where we assume $p(x|y)$ is distributed according to a multivariate normal distribution.

### 4.1.1 The Multivariate Normal Distribution
The multivariate normal (or Gaussian) distribution in $d$-dimensions is parameterized by a mean vector $\mu \in \mathbb{R}^d$ and a covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$, where $\Sigma \ge 0$ is symmetric and positive semi-definite. Written as $\mathcal{N}(\mu, \Sigma)$, its density is:
$$ p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu) \right) $$
Where $|\Sigma|$ denotes the determinant of $\Sigma$.

For a random variable $X \sim \mathcal{N}(\mu, \Sigma)$:
- **Mean**: $\text{E}[X] = \int_x x p(x; \mu, \Sigma)dx = \mu$
- **Covariance**: $\text{Cov}(Z) = \text{E}[(Z - \text{E}[Z])(Z - \text{E}[Z])^T] = \text{E}[ZZ^T] - (\text{E}[Z])(\text{E}[Z])^T$. For $X$, $\text{Cov}(X) = \Sigma$.

**Diagram Descriptions & Properties:**
1. **Varying variance (diagonal of $\Sigma$)**: The standard normal distribution has $\mu = 0$ and $\Sigma = I$. If we scale $\Sigma$ (e.g., $0.6I$ or $2I$), the Gaussian becomes more "compressed" or more "spread-out" respectively.
2. **Varying covariance (off-diagonal of $\Sigma$)**: Adding positive off-diagonal elements compresses the density towards the $45^\circ$ line ($x_1 = x_2$). Negative off-diagonal elements compress the density in the opposite direction. More generally, contours form ellipses when parameters vary.
3. **Varying mean ($\mu$)**: Changing $\mu$ simply shifts the center (mean) of the density around in the 2D coordinate space.

### 4.1.2 The Gaussian Discriminant Analysis Model
For continuous-valued random variables $x$, the GDA model assumes:
$$ y \sim \text{Bernoulli}(\phi) $$
$$ x | y = 0 \sim \mathcal{N}(\mu_0, \Sigma) $$
$$ x | y = 1 \sim \mathcal{N}(\mu_1, \Sigma) $$

Writing out the distributions:
$$ p(y) = \phi^y (1 - \phi)^{1-y} $$
$$ p(x|y=0) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2}(x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right) $$
$$ p(x|y=1) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2}(x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right) $$
*(Note: GDA typically uses two different mean vectors $\mu_0, \mu_1$ but a single shared covariance matrix $\Sigma$.)*

The log-likelihood of the data is:
$$ \ell(\phi, \mu_0, \mu_1, \Sigma) = \log \prod_{i=1}^n p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) = \log \prod_{i=1}^n p(x^{(i)} | y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi) $$

Maximizing $\ell$ yields the maximum likelihood estimates for the parameters:
$$ \phi = \frac{1}{n} \sum_{i=1}^n \mathbb{1}\{y^{(i)} = 1\} $$
$$ \mu_0 = \frac{\sum_{i=1}^n \mathbb{1}\{y^{(i)} = 0\}x^{(i)}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)} = 0\}} $$
$$ \mu_1 = \frac{\sum_{i=1}^n \mathbb{1}\{y^{(i)} = 1\}x^{(i)}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)} = 1\}} $$
$$ \Sigma = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T $$

**Diagram Description (GDA Decision Boundary)**: 
Pictorially, GDA fits two Gaussian distributions to the data of the two classes. They have the same shape and orientation (since they share $\Sigma$) but different means $\mu_0, \mu_1$. The decision boundary, where $p(y=1|x) = 0.5$, is a straight line separating the space.

### 4.1.3 Discussion: GDA and Logistic Regression
GDA has an interesting relationship with logistic regression. The probability $p(y=1|x; \phi, \mu_0, \mu_1, \Sigma)$ can be expressed as:
$$ p(y=1|x; \phi, \Sigma, \mu_0, \mu_1) = \frac{1}{1 + \exp(-\theta^T x)} $$
where $\theta$ is an appropriate function of $\phi, \Sigma, \mu_0, \mu_1$. This is identically the functional form that logistic regression uses to model $p(y=1|x)$.

**Comparison:**
- If $p(x|y)$ is multivariate Gaussian (with shared $\Sigma$), then $p(y|x)$ necessarily follows a logistic function.
- The converse is **not** true: $p(y|x)$ being a logistic function does not imply $p(x|y)$ is multivariate Gaussian. 
- **GDA makes stronger modeling assumptions** about the data. If these assumptions are true, GDA is *asymptotically efficient* (in the limit of large datasets, no algorithm is strictly better) and expects to learn better even with less data.
- **Logistic regression makes weaker assumptions** and is significantly more robust to deviations from modeling assumptions (e.g., if data is Poisson instead of Gaussian). For this reason, logistic regression is used more often in practice.

---

## 4.2 Naive Bayes (Option Reading)
For discrete-valued features (e.g., text classification and spam filtering), we can use the Naive Bayes algorithm.

**Feature Representation:**
An email is represented via a feature vector $x \in \{0, 1\}^{|V|}$ where $|V|$ is the size of the dictionary (vocabulary). $x_j = 1$ if the $j$-th dictionary word appears in the email, and $x_j = 0$ otherwise.

**The Naive Bayes (NB) Assumption:**
Modeling $p(x|y)$ explicitly for a 50,000-word dictionary would require $2^{50000}-1$ parameters, which is intractable. Instead, we make a strong assumption: the features $x_i$ are conditionally independent given $y$. 
*Note: This means $p(x_{2087}|y) = p(x_{2087}|y, x_{39831})$. This is not marginal independence, but conditional independence given the class $y$.*

Using the NB assumption, the joint probability simplifies:
$$ p(x_1, \dots, x_{50000} | y) = \prod_{j=1}^d p(x_j | y) $$

**Model Parameters & Maximum Likelihood Estimates:**
The model parameters are $\phi_{j|y=1} = p(x_j=1|y=1)$, $\phi_{j|y=0} = p(x_j=1|y=0)$, and $\phi_y = p(y=1)$.
The joint likelihood is:
$$ \mathcal{L}(\phi_y, \phi_{j|y=0}, \phi_{j|y=1}) = \prod_{i=1}^n p(x^{(i)}, y^{(i)}) $$

Maximizing yields:
$$ \phi_{j|y=1} = \frac{\sum_{i=1}^n \mathbb{1}\{x_j^{(i)} = 1 \land y^{(i)} = 1\}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)} = 1\}} $$
$$ \phi_{j|y=0} = \frac{\sum_{i=1}^n \mathbb{1}\{x_j^{(i)} = 1 \land y^{(i)} = 0\}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)} = 0\}} $$
$$ \phi_y = \frac{\sum_{i=1}^n \mathbb{1}\{y^{(i)} = 1\}}{n} $$

**Prediction:**
For a new example $x$:
$$ p(y=1|x) = \frac{\left( \prod_{j=1}^d p(x_j|y=1) \right) p(y=1)}{\left( \prod_{j=1}^d p(x_j|y=1) \right) p(y=1) + \left( \prod_{j=1}^d p(x_j|y=0) \right) p(y=0)} $$
We pick the class with the higher posterior probability.

*Extension*: Naive Bayes naturally extends to multinomial $x_j$ (features taking values in $\{1, \dots, k_j\}$). Continuous values (e.g., living area) can also be discretized into categorical buckets to apply Naive Bayes.

### 4.2.1 Laplace Smoothing
A critical flaw in basic Naive Bayes: if a word is never seen in the training data for a certain class, its maximum likelihood probability estimate will be 0. When making predictions, multiplying by $p(x_j|y) = 0$ will result in a $0/0$ probability, causing the algorithm to fail.

To fix this, we use **Laplace Smoothing**. For a multinomial variable $z \in \{1, \dots, k\}$, the standard estimate $\phi_j = \frac{\sum_{i=1}^n \mathbb{1}\{z^{(i)} = j\}}{n}$ is smoothed to:
$$ \phi_j = \frac{1 + \sum_{i=1}^n \mathbb{1}\{z^{(i)} = j\}}{k + n} $$
This ensures $\sum_{j=1}^k \phi_j = 1$ and $\phi_j \neq 0$ always.

Applying Laplace smoothing to Naive Bayes:
$$ \phi_{j|y=1} = \frac{1 + \sum_{i=1}^n \mathbb{1}\{x_j^{(i)} = 1 \land y^{(i)} = 1\}}{2 + \sum_{i=1}^n \mathbb{1}\{y^{(i)} = 1\}} $$
$$ \phi_{j|y=0} = \frac{1 + \sum_{i=1}^n \mathbb{1}\{x_j^{(i)} = 1 \land y^{(i)} = 0\}}{2 + \sum_{i=1}^n \mathbb{1}\{y^{(i)} = 0\}} $$

### 4.2.2 Event Models for Text Classification
- **Bernoulli Event Model** (used above): First, the class $y$ is chosen. Then, the sender runs through the dictionary and independently decides whether to include each word $j$ based on $p(x_j = 1 | y)$. Probability of a message is $p(y)\prod_{j=1}^d p(x_j|y)$.
- **Multinomial Event Model** (better for text): Let $x_j \in \{1, \dots, |V|\}$ denote the identity of the $j$-th word in an email of length $d$. A message is generated by first determining the class $y$. Then, each word $x_j$ is sampled independently from the same multinomial distribution over the vocabulary $p(x_j | y)$. Overall probability is $p(y)\prod_{j=1}^d p(x_j|y)$ but $x_j|y$ is now a multinomial distribution.

**Parameters for Multinomial Event Model:**
$\phi_y = p(y)$, $\phi_{k|y=1} = p(x_j=k|y=1)$, and $\phi_{k|y=0} = p(x_j=k|y=0)$. We assume $p(x_j|y)$ is independent of word position $j$.

The likelihood is:
$$ \mathcal{L}(\phi_y, \phi_{k|y=0}, \phi_{k|y=1}) = \prod_{i=1}^n \left( \prod_{j=1}^{d_i} p(x_j^{(i)} | y; \phi_{k|y=0}, \phi_{k|y=1}) \right) p(y^{(i)}; \phi_y) $$

With Laplace smoothing, the maximum likelihood estimates are:
$$ \phi_{k|y=1} = \frac{1 + \sum_{i=1}^n \sum_{j=1}^{d_i} \mathbb{1}\{x_j^{(i)} = k \land y^{(i)} = 1\}}{|V| + \sum_{i=1}^n \mathbb{1}\{y^{(i)} = 1\}d_i} $$
$$ \phi_{k|y=0} = \frac{1 + \sum_{i=1}^n \sum_{j=1}^{d_i} \mathbb{1}\{x_j^{(i)} = k \land y^{(i)} = 0\}}{|V| + \sum_{i=1}^n \mathbb{1}\{y^{(i)} = 0\}d_i} $$

Despite its simplicity and strong assumptions, Naive Bayes often works surprisingly well and is an excellent baseline classifier.