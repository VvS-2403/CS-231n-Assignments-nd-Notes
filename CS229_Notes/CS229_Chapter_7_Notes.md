# Chapter 7: Deep Learning

## 7.1 Supervised Learning with Non-linear Models
In supervised learning, we predict $y$ from input $x$ using a model $h_\theta(x)$. Previous models like linear regression ($h_\theta(x) = \theta^\top x$) and models with feature maps ($h_\theta(x) = \theta^\top \phi(x)$) were linear in the parameters $\theta$. Deep learning focuses on learning general families of models that are non-linear in **both** the parameters $\theta$ and inputs $x$, primarily neural networks.

Suppose we have training examples $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$. We define specific nonlinear models and cost functions:

### Regression Problems
For real-valued outputs $y^{(i)} \in \mathbb{R}$ and predictions $h_\theta(x) \in \mathbb{R}$, we use the **least squares cost function** for the $i$-th example:
$$ J^{(i)}(\theta) = \frac{1}{2}(h_\theta(x^{(i)}) - y^{(i)})^2 $$
The mean-square cost function for the entire dataset is:
$$ J(\theta) = \frac{1}{n}\sum_{i=1}^n J^{(i)}(\theta) $$
*(Note: Multiplying the cost function by a scalar like $1/n$ does not change its local or global minima).*

### Binary Classification
For inputs $x \in \mathbb{R}^d$, we define a parameterized model $\bar{h}_\theta : \mathbb{R}^d \to \mathbb{R}$ returning a **logit**, $\bar{h}_\theta(x)$. We apply the logistic function $g(\cdot)$ to output a probability $h_\theta(x) \in [0, 1]$:
$$ h_\theta(x) = g(\bar{h}_\theta(x)) = \frac{1}{1 + \exp(-\bar{h}_\theta(x))} $$
The conditional distribution is modeled as:
$$ P(y=1 | x; \theta) = h_\theta(x) \quad \text{and} \quad P(y=0 | x; \theta) = 1 - h_\theta(x) $$
The negative log-likelihood loss for a single example is the **logistic loss**:
$$ J^{(i)}(\theta) = -\log p(y^{(i)} | x^{(i)}; \theta) = \ell_{\text{logistic}}(\bar{h}_\theta(x^{(i)}), y^{(i)}) $$
Total loss: $J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta)$.

### Multi-class Classification
For $y \in \{1, 2, \dots, k\}$, the model outputs a $k$-dimensional vector of logits, $\bar{h}_\theta(x) \in \mathbb{R}^k$. The **softmax function** converts this into a probability vector:
$$ P(y=j | x; \theta) = \frac{\exp(\bar{h}_\theta(x)_j)}{\sum_{s=1}^k \exp(\bar{h}_\theta(x)_s)} $$
The negative log-likelihood loss for a single example is the **cross-entropy loss**:
$$ J^{(i)}(\theta) = -\log p(y^{(i)} | x^{(i)}; \theta) = -\log \left( \frac{\exp(\bar{h}_\theta(x^{(i)})_{y^{(i)}})}{\sum_{s=1}^k \exp(\bar{h}_\theta(x^{(i)})_s)} \right) = \ell_{\text{ce}}(\bar{h}_\theta(x^{(i)}), y^{(i)}) $$

### Optimizers (SGD)
Loss functions $J(\theta)$ are typically optimized using Gradient Descent (GD) or Stochastic Gradient Descent (SGD).
**Gradient Descent (GD) Update Rule:**
$$ \theta := \theta - \alpha \nabla_\theta J(\theta) $$
where $\alpha > 0$ is the learning rate.

**Algorithm 1: Stochastic Gradient Descent (SGD)**
1. Initialize $\theta$ randomly.
2. For $i = 1$ to $n_{\text{iter}}$:
   - Sample $j$ uniformly from $\{1, \dots, n\}$ and update $\theta$:
   $$ \theta := \theta - \alpha \nabla_\theta J^{(j)}(\theta) $$

**Algorithm 2: Mini-batch Stochastic Gradient Descent**
For faster execution via hardware parallelization, Mini-batch SGD computes gradients for $B$ examples simultaneously.
1. Initialize $\theta$ randomly.
2. For $i = 1$ to $n_{\text{iter}}$:
   - Sample $B$ examples $j_1, \dots, j_B$ uniformly from $\{1, \dots, n\}$ without replacement, and update:
   $$ \theta := \theta - \frac{\alpha}{B} \sum_{k=1}^B \nabla_\theta J^{(j_k)}(\theta) $$

Typical deep learning workflow: 
1. Define neural network parametrization $h_\theta(x)$.
2. Use backpropagation to efficiently compute gradients $\nabla_\theta J^{(j)}(\theta)$.
3. Run SGD / mini-batch SGD.

---

## 7.2 Neural Networks
Neural networks represent parametrizations $\bar{h}_\theta(x)$ involving combinations of matrix multiplications and entry-wise non-linear operations.

### A Neural Network with a Single Neuron
To predict housing prices while strictly avoiding negative predictions, we want a "kink" in the function (Fig 7.1: Graph of housing price vs square feet, showing a flat zero price up to a point, then linearly increasing).
We use the simplest parameterization mapping $x \to y$:
$$ \bar{h}_\theta(x) = \max(wx + b, 0) $$
where $\theta = (w, b) \in \mathbb{R}^2$. The function $\max\{t, 0\}$ is the **ReLU (Rectified Linear Unit)**, an **activation function**.
For multi-dimensional input $x \in \mathbb{R}^d$, this 1-layer neural network is:
$$ \bar{h}_\theta(x) = \text{ReLU}(w^\top x + b) $$
where $w \in \mathbb{R}^d$ (weight vector) and $b \in \mathbb{R}$ (bias).

### Stacking Neurons
We can stack neurons to build complex structures.
*Figure 7.2 Description*: Diagram showing a small neural network for housing prices. Input features (Size, # Bedrooms, Zip Code, Wealth) are fed into intermediate variables (Family Size, Walkable, School Quality), which in turn predict the final Price $y$.

Intermediate variables $a_1, a_2, a_3$ (hidden units) are parameterized as single-neuron networks:
$$ a_1 = \text{ReLU}(\theta_1 x_1 + \theta_2 x_2 + \theta_3) $$
$$ a_2 = \text{ReLU}(\theta_4 x_3 + \theta_5) $$
$$ a_3 = \text{ReLU}(\theta_6 x_3 + \theta_7 x_4 + \theta_8) $$
Final output combines them linearly:
$$ \bar{h}_\theta(x) = \theta_9 a_1 + \theta_{10} a_2 + \theta_{11} a_3 + \theta_{12} $$
*Note: Artificial neural networks are inspired by biological ones, where hidden units represent neurons and parameters represent synapses.*

### Two-layer Fully-Connected Neural Networks
To make the model generic without relying on prior knowledge (like manually matching inputs to hidden units), we use a **fully-connected** structure where each intermediate variable depends on all inputs.
For $m$ hidden units and $x \in \mathbb{R}^d$:
$$ z_j = (w_j^{[1]})^\top x + b_j^{[1]}, \quad a_j = \text{ReLU}(z_j) \quad \forall j \in \{1, \dots, m\} $$
Output:
$$ \bar{h}_\theta(x) = (w^{[2]})^\top a + b^{[2]} $$

### Vectorization
To leverage GPU parallelism and optimized BLAS libraries, we avoid `for` loops by using matrix algebra.
Define weight matrix $W^{[1]} \in \mathbb{R}^{m \times d}$ by stacking row vectors $(w_j^{[1]})^\top$, and $b^{[1]} \in \mathbb{R}^{m \times 1}$. We can succinctly write:
$$ z = W^{[1]}x + b^{[1]} $$
$$ a = \text{ReLU}(z) $$
$$ \bar{h}_\theta(x) = W^{[2]}a + b^{[2]} $$
The collection of $W^{[1]}, b^{[1]}$ is the **first layer**, and $W^{[2]}, b^{[2]}$ is the **second layer**. Vector $a$ is the **hidden layer**.

### Multi-layer Fully-Connected Neural Networks (MLP)
For $r$ layers, with weight matrices $W^{[1]}, \dots, W^{[r]}$ and biases $b^{[1]}, \dots, b^{[r]}$:
$$ a^{[1]} = \text{ReLU}(W^{[1]}x + b^{[1]}) $$
$$ \dots $$
$$ a^{[r-1]} = \text{ReLU}(W^{[r-1]}a^{[r-2]} + b^{[r-1]}) $$
$$ \bar{h}_\theta(x) = W^{[r]}a^{[r-1]} + b^{[r]} $$
Often, the final layer doesn't use an activation function (is linear). The recursive form (setting $a^{[0]} = x$):
$$ a^{[k]} = \text{ReLU}(W^{[k]}a^{[k-1]} + b^{[k]}) \quad \forall k = 1, \dots, r-1 $$

### Other Activation Functions
ReLU can be replaced by other non-linear mapping functions $\sigma : \mathbb{R} \to \mathbb{R}$:
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Tanh**: $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
- **Leaky ReLU**: $\sigma(z) = \max\{z, \gamma z\}, \gamma \in (0, 1)$
- **GELU**: $\sigma(z) = \frac{z}{2} \left(1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right)$
- **Softplus**: $\sigma(z) = \frac{1}{\beta} \log(1 + \exp(\beta z)), \beta > 0$

*Figure 7.3 Description*: Plots of these activation functions. Sigmoid and tanh bounds are restricted and gradients vanish at infinities, so they are less used today. Softplus is a smooth ReLU with a proper second derivative. GELU (common in NLP models like BERT/GPT) and Leaky ReLU maintain non-zero gradients for negative inputs.

**Why not the identity function $\sigma(z) = z$?**
If activations are linear, the entire network collapses into a single linear transformation: $W^{[2]}W^{[1]}x = \tilde{W}x$. You lose all non-linear representational power, reducing the model to simple linear regression.

### Connection to the Kernel Method
Instead of manually crafting feature maps $\phi(x)$ (feature engineering), Deep Learning automatically learns the right feature representation. The output $\bar{h}_\theta(x) = W^{[r]}\phi_\beta(x) + b^{[r]}$, where $\phi_\beta(x)$ equals $a^{[r-1]}$. The network optimizes both the linear final layer and the intermediate "feature map" representations ($\beta$), making deep learning highly flexible (but inherently a "black box" as features are hard to interpret).

---

## 7.3 Modules in Modern Neural Networks
An MLP can be represented as composed operations. We define a matrix multiplication module $\text{MM}_{W,b}(z) = Wz + b$.
$$ \text{MLP}(x) = \text{MM}(\sigma(\text{MM}(\dots \text{MM}(x)))) $$

*Figure 7.4 Description*: (Left) Diagram of an MLP with $r$ layers comprising alternating MM and $\sigma$ blocks. (Right) A residual network composed of multiple Res blocks, each maintaining a skip connection around MM and $\sigma$ blocks.

### Residual Connections
A prominent architecture feature (used in ResNet, Transformers) is the **residual block**:
$$ \text{Res}(z) = z + \sigma(\text{MM}(\sigma(\text{MM}(z)))) $$
A simplified ResNet is:
$$ \text{ResNet-S}(x) = \text{MM}(\text{Res}(\text{Res}(\dots \text{Res}(x)))) $$

### Layer Normalization (LN)
Commonly used post-activation, mapping vector $z \in \mathbb{R}^m$ to a normalized $\text{LN}(z) \in \mathbb{R}^m$.
First, the standard normalization sub-module $\text{LN-S}(z)$:
$$ \text{LN-S}(z)_i = \frac{z_i - \hat{\mu}}{\hat{\sigma}} $$
where empirical mean $\hat{\mu} = \frac{1}{m}\sum_{i=1}^m z_i$ and empirical std dev $\hat{\sigma} = \sqrt{\frac{1}{m}\sum_{i=1}^m (z_i - \hat{\mu})^2}$. (Note: divided by $m$, not $m-1$, ensuring output sum-of-squares is 1).

Full Layer Normalization includes learnable scalar parameters $\beta$ (desired mean) and $\gamma$ (desired standard deviation):
$$ \text{LN}(z) = \beta + \gamma \cdot \text{LN-S}(z) $$

**Scaling-invariant property:** 
If we compose LN with MM, scaling the parameters $W, b$ by $\alpha > 0$ does not change the output:
$$ \text{LN}(\text{MM}_{\alpha W, \alpha b}(z)) = \text{LN}(\text{MM}_{W, b}(z)) $$
This stabilizes deep networks against weight magnitude scaling. 

*Other normalizations include Batch Normalization and Group Normalization (common in computer vision, whereas LN is popular in NLP).*

### Convolutional Layers
Used heavily in Computer Vision (CNNs) and NLP, they rely on parameter sharing to drastically reduce compute/memory compared to standard MM.

**Simplified 1D Convolution ($\text{Conv1D-S}$)**:
Filter vector $w \in \mathbb{R}^k$ (filter size $k = 2\ell + 1$). Zero-pad input $z$ by $\ell$ on both sides. Output is a linear combination of subsets of $z_j$:
$$ \text{Conv1D-S}(z)_i = \sum_{j=1}^{2\ell+1} w_j z_{i-\ell+(j-1)} $$
It can be viewed as a sparse matrix multiplication $Qz$, where $Q \in \mathbb{R}^{m \times m}$ forms a banded diagonal matrix with shared weights. Complexity is $\mathcal{O}(km)$ rather than $\mathcal{O}(m^2)$.

**Multi-channel 1D Convolution**:
For an input with $C$ channels ($z \in \mathbb{R}^{m \times C}$) and output with $C'$ channels, the operation involves $C \times C'$ filters:
$$ \text{Conv1D}(z)_i = \sum_{j=1}^C \text{Conv1D-S}_{i,j}(z_j) $$
Total parameters = $kCC'$.

**2D Convolution ($\text{Conv2D}$)**:
Applies a $k \times k$ filter over a 2D spatial grid. For $C$ input channels and $C'$ output channels, parameters are organized as a 4D tensor $C \times C' \times k \times k$. Total parameters = $C C' k^2$.

---

## 7.4 Backpropagation
**Theorem 7.4.1 (Informal)**: If a real-valued function $f: \mathbb{R}^\ell \to \mathbb{R}$ is computed by a differentiable circuit of size $N$, its gradient $\nabla f$ can be computed in $\mathcal{O}(N)$ time by a circuit of size $\mathcal{O}(N)$.
This proves we can compute gradients ($\nabla J^{(j)}(\theta)$) roughly as fast as the forward evaluation itself.

### 7.4.1 Preliminaries on Partial Derivatives
For $z \in \mathbb{R}^m, u = g(z) \in \mathbb{R}^n, J = f(u) \in \mathbb{R}$, the multivariable chain rule is:
$$ \frac{\partial J}{\partial z_i} = \sum_{j=1}^n \frac{\partial J}{\partial u_j} \cdot \frac{\partial g_j}{\partial z_i} $$
In vector form:
$$ \frac{\partial J}{\partial z} = \text{Jacobian}^\top \cdot \frac{\partial J}{\partial u} $$
*(Note: $\frac{\partial J}{\partial z}$ has the same dimensions as $z$. For matrices $z \in \mathbb{R}^{r \times s}$, we just use indices $\frac{\partial J}{\partial z_{ik}} = \sum_j \frac{\partial J}{\partial u_j} \frac{\partial g_j}{\partial z_{ik}}$).*

We modularize this mapping into a **backward function** $\mathcal{B}[g, z]$ that maps $\frac{\partial J}{\partial u} \to \frac{\partial J}{\partial z}$:
$$ \frac{\partial J}{\partial z} = \mathcal{B}[g, z]\left( \frac{\partial J}{\partial u} \right) $$

### 7.4.2 General Strategy of Backpropagation
A loss function is a composition of modules: $J = M_k(M_{k-1}(\dots M_1(x)))$. 
Let intermediate states be $u^{[0]}=x, u^{[1]} = M_1(u^{[0]}), \dots, J = u^{[k]}$.

1. **Forward pass**: Compute and store $u^{[1]}, \dots, u^{[k]}$ sequentially.
2. **Backward pass**: Compute $\frac{\partial J}{\partial u^{[i]}}$ in reverse order, then compute parameter gradients $\frac{\partial J}{\partial \theta^{[i]}}$.
   - Gradient w.r.t input of module: $\frac{\partial J}{\partial u^{[i-1]}} = \mathcal{B}[M_i, u^{[i-1]}]\left( \frac{\partial J}{\partial u^{[i]}} \right)$
   - Gradient w.r.t parameters: $\frac{\partial J}{\partial \theta^{[i]}} = \mathcal{B}[M_i, \theta^{[i]}]\left( \frac{\partial J}{\partial u^{[i]}} \right)$

*Figure 7.5 Description*: A flowchart showing the Forward Pass evaluating $M_i(u^{[i-1]}) = u^{[i]}$, and the Backward Pass computing gradients $\frac{\partial J}{\partial u^{[i-1]}}$ and $\frac{\partial J}{\partial \theta^{[i]}}$ via $\mathcal{B}$.

### 7.4.3 Backward Functions for Basic Modules
**Backward function for Matrix Multiplication (MM)**:
For $u = Wz + b$ (where $z \in \mathbb{R}^m, W \in \mathbb{R}^{n \times m}$):
- W.r.t $z$: $\mathcal{B}[\text{MM}, z](v) = W^\top v \in \mathbb{R}^m$
- W.r.t $W$: $\mathcal{B}[\text{MM}, W](v) = v z^\top \in \mathbb{R}^{n \times m}$
- W.r.t $b$: $\mathcal{B}[\text{MM}, b](v) = v \in \mathbb{R}^n$
Computations take $\mathcal{O}(mn)$ time, matching forward pass complexity.

**Backward function for Activations**:
For $u = \sigma(z)$:
- $\mathcal{B}[\sigma, z](v) = \text{diag}(\sigma'(z_1), \dots, \sigma'(z_m))v = \sigma'(z) \odot v \in \mathbb{R}^m$
Where $\odot$ is the element-wise product. Computable in $\mathcal{O}(m)$ time.

**Backward function for Loss functions**:
For scalar outputs, $v=1$.
- **Squared loss** ($\ell_{\text{MSE}}$): $\mathcal{B}[\ell_{\text{MSE}}, z](v) = (z - y) \cdot v$
- **Logistic loss**: $\mathcal{B}[\ell_{\text{logistic}}, t](v) = \left( \frac{1}{1 + \exp(-t)} - y \right) \cdot v$
- **Cross-entropy loss**: $\mathcal{B}[\ell_{\text{ce}}, t](v) = (\text{softmax}(t) - e_y) \cdot v$, where $e_y$ is a one-hot vector for class $y$.

### 7.4.4 Back-propagation for MLPs
**Algorithm 3: Back-propagation for multi-layer neural networks**
1. **Forward pass**: Compute and store $a^{[k]}, z^{[k]}$, and $J$. 
   $z^{[k]} = W^{[k]}a^{[k-1]} + b^{[k]}$ and $a^{[k]} = \sigma(z^{[k]})$.
2. **Backward pass**: Initialize with $\frac{\partial J}{\partial z^{[r]}} = \frac{1}{1 + \exp(-z^{[r]})} - y$ (assuming logistic loss).
3. **For $k = r-1$ down to 0**:
   - Compute gradient w.r.t parameters:
     $$ \frac{\partial J}{\partial W^{[k+1]}} = \frac{\partial J}{\partial z^{[k+1]}} (a^{[k]})^\top $$
     $$ \frac{\partial J}{\partial b^{[k+1]}} = \frac{\partial J}{\partial z^{[k+1]}} $$
   - If $k \geq 1$, compute gradient w.r.t hidden inputs:
     $$ \frac{\partial J}{\partial a^{[k]}} = (W^{[k+1]})^\top \frac{\partial J}{\partial z^{[k+1]}} $$
     $$ \frac{\partial J}{\partial z^{[k]}} = \sigma'(z^{[k]}) \odot \frac{\partial J}{\partial a^{[k]}} $$

---

## 7.5 Vectorization Over Training Examples
To process multiple examples in parallel, we stack them into matrices.

### The Basic Idea
For examples $x^{(1)}, x^{(2)}, x^{(3)}$, we vectorize $z^{[1]} = W^{[1]}x + b^{[1]}$ by stacking inputs into columns of $X \in \mathbb{R}^{d \times 3}$:
$$ X = \begin{bmatrix} | & | & | \\ x^{(1)} & x^{(2)} & x^{(3)} \\ | & | & | \end{bmatrix} $$
$$ Z^{[1]} = W^{[1]}X + b^{[1]} $$
*(Note: $b^{[1]}$ is implicitly broadcasted across the columns).*

### Complications/Subtlety in the Implementation
While theoretical math typically treats data as **column vectors**, most deep learning libraries (PyTorch, TensorFlow, etc.) structure data matrices in **row-major format** (examples are rows).
To bridge this:
- Data matrix $X \in \mathbb{R}^{3 \times d}$
- First layer weights transposed $W^{[1]} \in \mathbb{R}^{d \times m}$
- Bias $b^{[1]} \in \mathbb{R}^{1 \times m}$
The matrix multiplication order is flipped:
$$ Z^{[1]} = XW^{[1]} + b^{[1]} \in \mathbb{R}^{3 \times m} $$