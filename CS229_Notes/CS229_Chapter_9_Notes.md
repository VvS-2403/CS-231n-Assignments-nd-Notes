# CS229: Chapter 9 - Regularization and Model Selection

## 9.1 Regularization
Overfitting is typically a result of using overly complex models. To achieve the optimal **bias-variance tradeoff**[^5], we must choose a proper model complexity. When complexity is measured by the number of parameters, we can vary the size of the model (e.g., the width of a neural net). However, model complexity can also be a function of the parameters (e.g., the $\ell_2$ norm of the parameters), which may not necessarily depend on the number of parameters. In such cases, we use **regularization** to control model complexity and prevent overfitting.

Regularization involves adding an additional term, the **regularizer** $R(\theta)$, to the training loss/cost function[^1]:
$$J_\lambda(\theta) = J(\theta) + \lambda R(\theta) \quad \text{(9.1)}$$
- $J_\lambda$: Regularized loss
- $\lambda \ge 0$: Regularization parameter (controls the balance between fitting the data and minimizing complexity)
- $R(\theta)$: Regularizer (nonnegative function, typically a measure of model complexity. Can sometimes depend on the training dataset in modern approaches)

When $\lambda = 0$, it is equivalent to the original loss. When $\lambda$ is extremely large, the original loss becomes ineffective, likely leading to high bias.

### $\ell_2$ Regularization (Weight Decay)
The most commonly used regularizer is $\ell_2$ regularization:
$$R(\theta) = \frac{1}{2}\|\theta\|^2_2$$
It encourages the optimizer to find a model with a small $\ell_2$ norm. In deep learning, it is often called **weight decay**, because gradient descent with learning rate $\eta$ on the regularized loss is equivalent to shrinking/decaying $\theta$ by a scalar factor of $1 - \lambda\eta$ before applying the standard gradient:
$$\theta \gets \theta - \eta\nabla J_\lambda(\theta) = \theta - \eta\lambda\theta - \eta\nabla J(\theta) = \underbrace{(1 - \lambda\eta)\theta}_{\text{decaying weights}} - \eta\nabla J(\theta) \quad \text{(9.2)}$$

### Imposing Inductive Biases and Sparsity
Regularization can impose inductive biases or structures:
- **Sparsity**: A prior belief that the number of non-zeros in the ground-truth model parameters is small[^2]. We could penalize the number of non-zeros, denoted by $\|\theta\|_0$.
- Because $\|\theta\|_0$ is non-continuous and cannot be optimized with (stochastic) gradient descent, a common continuous surrogate is the $\ell_1$ norm (LASSO): $R(\theta) = \|\theta\|_1$[^3].
- Imposing structure narrows the search space (reduces model family complexity), leading to better generalization, but risks increasing bias if the true model doesn't match the structure.

**Regularization in other domains:**
- **Kernel Methods:** $\ell_2$ norm is commonly used because $\ell_1$ is typically not compatible with the kernel trick (optimal solution cannot be written as functions of inner products).
- **Deep Learning:** Regularizers include $\ell_2$ regularization, dropout, data augmentation, spectral norm regularization, and Lipschitzness regularization.

---

## 9.2 Implicit Regularization Effect (Optional)
**Implicit regularization** (or implicit bias / algorithmic regularization) refers to the phenomenon where optimizers implicitly impose structures on parameters beyond the regularized loss.

In deep learning, the loss often has multiple (approximate) global minima. Although these minima have similar training losses, their generalization performance can dramatically differ. 
- Example: One global minimum might yield a much more Lipschitz or sparse model than others, thus having a better test error.
- Optimizers (or their components) bias towards finding global minima with certain properties.

**Diagram Descriptions:**
- **Figure 9.1:** An illustration plotting "loss" against $\theta$. It demonstrates that different global minima of the training loss can have wildly different test performances.
- **Figure 9.2:** 
  - *Left:* Shows the performance of neural networks trained with two different learning rate schedules on the CIFAR-10 dataset. Both fit the training data perfectly (same regularized loss) but have vastly different generalization performances.
  - *Right:* Shows that on a synthetic dataset, optimizers with different initializations achieve the same training error but completely different generalization performance[^4].

**Takeaway:** The choice of optimizer affects generalization!
**Heuristics:** Optimizers can be biased towards more generalizable solutions using:
- Larger initial learning rate
- Smaller initialization
- Smaller batch size
- Momentum

*Conjecture:* Stochasticity helps optimizers find **flatter global minima** (where curvature of the loss is small), which tend to give more Lipschitz models and better generalization.

---

## 9.3 Model Selection via Cross Validation
How do we automatically select a model that represents a good bias-variance tradeoff (e.g., choosing polynomial degree $k$, or SVM parameter $C$)?

Let $\mathcal{M} = \{M_1, \dots, M_d\}$ be a finite set of models[^6]. *Empirical risk minimization (picking the hypothesis with the lowest training error) fails because it always selects a high-variance, complex model.*

### Hold-out Cross Validation (Simple Cross Validation)
1. Randomly split dataset $S$ into $S_{\text{train}}$ (e.g., 70%) and a hold-out cross validation set $S_{\text{cv}}$ (30%). (The size depends on the total examples; for large sets like ImageNet, 5% may suffice).
2. Train each model $M_i$ on $S_{\text{train}}$ to get hypothesis $h_i$.
3. Select and output $h_i$ with the smallest validation error $\hat{\varepsilon}_{S_{\text{cv}}}(h_i)$.

*Optional:* Retrain the selected model $M_i$ on the entire training set $S$ (except for models very sensitive to initial conditions/data perturbations).
*Drawback:* "Wastes" a portion of the data (e.g., 30%), which is detrimental when data is scarce (e.g., $n=20$).

### $k$-fold Cross Validation
Holds out less data to alleviate data scarcity:
1. Randomly split $S$ into $k$ disjoint subsets $S_1, \dots, S_k$ (each of size $m/k$).
2. For each model $M_i$:
   - For $j = 1, \dots, k$:
     - Train $M_i$ on all data except $S_j$ to get hypothesis $h_{ij}$.
     - Test $h_{ij}$ on $S_j$ to get error $\hat{\varepsilon}_{S_j}(h_{ij})$.
   - The estimated generalization error of $M_i$ is the average of the $\hat{\varepsilon}_{S_j}(h_{ij})$ values.
3. Pick the model $M_i$ with the lowest estimated generalization error, and retrain on the entire dataset $S$.

*Typical choice:* $k=10$.
*Computational Cost:* More expensive since each model is trained $k$ times.

### Leave-one-out Cross Validation
An extreme version of $k$-fold cross validation where $k=m$ (number of examples). Used when data is extremely scarce. We hold out one example at a time.

*(Note: Cross validation can also be used simply to evaluate a single model or algorithm).*

---

## 9.4 Bayesian Statistics and Regularization

### Frequentist vs. Bayesian View
- **Frequentist View:** $\theta$ is an unknown but constant parameter. We use tools like Maximum Likelihood Estimation (MLE):
  $$\theta_{\text{MLE}} = \arg \max_\theta \prod_{i=1}^n p(y^{(i)}|x^{(i)}; \theta)$$
- **Bayesian View:** $\theta$ is an unknown random variable. We specify a **prior distribution** $p(\theta)$ expressing our prior beliefs.

### Posterior Distribution
Given a training set $S = \{(x^{(i)}, y^{(i)})\}_{i=1}^n$, the posterior distribution on $\theta$ is computed using Bayes' rule[^7]:
$$p(\theta|S) = \frac{p(S|\theta)p(\theta)}{p(S)} = \frac{\left( \prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta) \right) p(\theta)}{\int_\theta \left( \prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta) p(\theta) \right) d\theta} \quad \text{(9.3)}$$

*(Example: For Bayesian logistic regression, $p(y^{(i)}|x^{(i)}, \theta) = h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}$, where $h_\theta(x^{(i)}) = 1/(1 + \exp(-\theta^T x^{(i)}))$.)*

### Fully Bayesian Prediction
To make a prediction on a new test example $x$, we compute the posterior distribution on the class label by averaging over $\theta$:
$$p(y|x, S) = \int_\theta p(y|x, \theta)p(\theta|S)d\theta \quad \text{(9.4)}$$
Expected value of $y$ given $x$[^8]:
$$E[y|x, S] = \int_y y p(y|x, S)dy$$
*Issue:* The integral over high-dimensional $\theta$ is computationally difficult and usually lacks a closed-form solution.

### Maximum A Posteriori (MAP) Estimate
To approximate the computationally expensive posterior, we replace it with a single point estimate called the **MAP estimate**:
$$\theta_{\text{MAP}} = \arg \max_\theta \prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)p(\theta) \quad \text{(9.5)}$$
*(Note: This is identical to MLE except for the addition of the prior term $p(\theta)$.)*

**Connection to Regularization:**
If we assume a Gaussian prior $p(\theta) \sim \mathcal{N}(0, \tau^2I)$, the fitted $\theta_{\text{MAP}}$ will have a smaller norm than the MLE estimate. This causes the Bayesian MAP estimate to be less susceptible to overfitting than MLE (effectively acting as $\ell_2$ regularization). For example, Bayesian logistic regression is highly effective in text classification, even when $d \gg n$.

---

## Footnotes
[^1]: Notations generally omit the dependency on the training dataset for simplicity—writing $J(\theta)$ even though it obviously depends on the training dataset.
[^2]: For linear models, sparsity means the model just uses a few coordinates of the inputs to make an accurate prediction.
[^3]: An intuition for $\|\theta\|_1$ as a surrogate for sparsity: assuming the parameter is on the unit sphere, the parameter with the smallest $\ell_1$ norm also happens to be the sparsest parameter with only 1 non-zero coordinate. Thus, sparsity and $\ell_1$ norm give the same extremal points.
[^4]: The setting is the same as in Woodworth et al. [2020], HaoChen et al. [2020].
[^5]: "Twin evils of bias and variance" (though they are very different beasts, perhaps "fraternal twin evils").
[^6]: If choosing from an infinite set of models (e.g., bandwidth $\tau \in \mathbb{R}^+$), we may discretize it to a finite set.
[^7]: Since we view $\theta$ as a random variable, it is okay to condition on its value and write $p(y|x, \theta)$ instead of $p(y|x; \theta)$.
[^8]: The integral is replaced by a summation if $y$ is discrete-valued.