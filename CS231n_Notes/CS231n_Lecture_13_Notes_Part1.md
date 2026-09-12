# CS231n Lecture 13: Generative Models (Part 1)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Overview](#overview)
- [Supervised vs Unsupervised Learning](#supervised-vs-unsupervised-learning)
- [Generative Models](#generative-models)
  - [Taxonomy of Generative Models](#taxonomy-of-generative-models)
- [PixelRNN and PixelCNN](#pixelrnn-and-pixelcnn)
  - [PixelRNN](#pixelrnn)
  - [PixelCNN](#pixelcnn)
- [Variational Autoencoders (VAE)](#variational-autoencoders-vae)
  - [Background: Autoencoders](#background-autoencoders)
  - [Variational Autoencoders](#variational-autoencoders-1)
  - [The ELBO Derivation](#the-elbo-derivation)
  - [VAE Training](#vae-training)
  - [Generating with VAEs](#generating-with-vaes)

---

## Overview

- This lecture explores Unsupervised Learning with a specific focus on Generative Models.

- Generative models represent a significant frontier in machine learning.
  - They allow computers not just to categorize data, but to deeply understand the fundamental distributions of the visual world.
  - This understanding enables the synthesis of entirely new, realistic examples.

- The three main classes of generative models discussed are:
  1. PixelRNN / PixelCNN
  2. Variational Autoencoders (VAEs)
  3. Generative Adversarial Networks (GANs)

## Supervised vs Unsupervised Learning

- To understand generative models, we must draw a clear distinction between supervised and unsupervised learning paradigms.

- **Supervised Learning**:
  - Data comes in pairs $(x, y)$, where $x$ is the input data (e.g., an image) and $y$ is the corresponding label or target output provided by a human.
  - The fundamental goal is to learn a mapping function from $x$ to $y$.
  - Examples include:
    - Image classification (mapping an image to a label like "Cat")
    - Object detection (drawing bounding boxes around objects)
    - Semantic segmentation (assigning a class label to every pixel)
    - Image captioning (mapping an image to a descriptive sentence)
  - The common thread is the reliance on a supervisory signal $y$.

- **Unsupervised Learning**:
  - Operates on data $x$ without any associated labels.
  - The primary goal is to discover and learn the underlying hidden structure of the data itself, rather than predicting a specific output.
  - It does not require human annotation, allowing the use of vast amounts of cheap, readily available data.
  - Examples include:
    - Clustering algorithms like K-means (grouping similar data points)
    - Dimensionality reduction techniques like PCA (projecting high-dimensional data into a lower-dimensional space)
    - Feature learning through autoencoders
    - Density estimation

> **Key Insight**: The holy grail of unsupervised learning is to understand the structure of the visual world. Because training data without labels is cheap and abundant, unsupervised learning offers a pathway to training massive models that encode a deep understanding of the world without being bottlenecked by the expensive process of human annotation.

## Generative Models

- Generative models are a core focus within unsupervised learning, specifically addressing the problem of density estimation.

- The fundamental objective:
  - Take a set of training data drawn from some true, underlying distribution $p_{data}(x)$.
  - Learn a model distribution $p_{model}(x)$ that closely approximates it.

- Once trained, we can generate completely new samples from $p_{model}(x)$ that are ideally indistinguishable from the original training data.

- Applications of generative models are numerous:
  - Producing realistic samples for artwork
  - Performing super-resolution to enhance image quality
  - Colorizing black-and-white photos
  - Simulating realistic environments using models of time-series data (highly valuable for planning and reinforcement learning)
  - Learning robust, latent representations of the data that can be used as general features for downstream supervised tasks

- Density estimation can be approached in two primary ways:
  1. **Explicit density estimation**: Involves designing a model that explicitly defines a tractable density function $p_{model}(x)$, which can be directly optimized using maximum likelihood.
  2. **Implicit density estimation**: Does not require explicitly defining the density function. Instead, it focuses on learning a model capable of generating samples drawn from $p_{model}(x)$ without ever computing the exact probability density.

### Taxonomy of Generative Models

- We can classify generative models into a full taxonomy based on how they handle density estimation.

- **Explicit Density Models**: The probability distribution is explicitly defined.
  - **Tractable Density Models**: Can directly and efficiently compute the exact probability of data.
    - Examples: Fully Visible Belief Nets like NADE, MADE, and PixelRNN/CNN.
  - **Approximate Density Models**: Define a density function that is computationally intractable to evaluate exactly, requiring approximations.
    - Examples: Variational Autoencoder (VAE) using variational inference, and Boltzmann Machines using Markov Chain Monte Carlo.

- **Implicit Density Models**: Forgo explicit definition of the probability distribution, focusing entirely on the sampling process.
  - **Direct Implicit Models**: Learn a direct mapping from a simple noise distribution to the data distribution.
    - Example: Generative Adversarial Networks (GANs).
  - **Markov Chain based Implicit Models**: Rely on Markov Chains for generation.
    - Example: Generative Stochastic Networks (GSNs).

## PixelRNN and PixelCNN

- PixelRNN and PixelCNN are Fully Visible Belief Networks, falling under explicit density models.

- They aim to compute the exact likelihood of an image directly.

- The core idea relies on the chain rule of probability:
  - We decompose the joint likelihood of all pixels in an image into a product of 1-dimensional conditional distributions:
  
  $$p(x) = \prod_{i=1}^n p(x_i | x_1, \dots, x_{i-1})$$

  - Here, $x$ is the entire image, $n$ is the total number of pixels, and $x_i$ represents the value of the $i$-th pixel.
  - $p(x_i | x_1, \dots, x_{i-1})$ is the probability of the $i$-th pixel given the values of all previously generated pixels.

- To make this sequence well-defined, we impose an arbitrary ordering on the pixels:
  - Typically, starting from the top-left corner and scanning row by row down to the bottom-right.

- The task is to model these complex conditional distributions, a perfect fit for neural networks.

- During training, we explicitly maximize the likelihood of the training data.

### PixelRNN

- PixelRNN utilizes a Recurrent Neural Network (RNN) — usually an LSTM — to model the dependency of each pixel on the previous pixels.

- The generation process:
  1. Starts at the top-left corner.
  2. The model predicts the first pixel, and its hidden state is updated.
  3. To predict the next pixel, the RNN uses the previously generated pixels and its internal state.

- Visually, generation moves diagonally across the image grid.
  - The state of each newly generated pixel depends on the pixels immediately above and to the left of it.
  - This creates a complex web of dependencies managed by the LSTM.

- **Drawback**: Speed.
  - Generation of each pixel strictly depends on the previous steps, so it cannot be parallelized.
  - Generating a high-resolution image pixel-by-pixel using a deep RNN is excruciatingly slow.

### PixelCNN

- PixelCNN improves upon PixelRNN by addressing computational inefficiencies.

- Instead of an RNN, it models the conditional distribution of a pixel using a Convolutional Neural Network (CNN) applied over a restricted context region.

- **Masked Convolutions**:
  - A convolution is centered on the current pixel being predicted.
  - To prevent "cheating" by looking at future pixels, the convolutional filters are masked.
  - This zeroes out the weights for the current pixel and any pixels that appear later in the ordering.
  - Visually, it looks like a standard square filter where the bottom half and the right half of the center row are blanked out.

- **Training Advantage**:
  - Since we know the true values of all pixels in a training image, the masked convolutions for every pixel can be computed in parallel in a single forward pass.
  - This makes training significantly faster than PixelRNN.

- **Testing Disadvantage**:
  - When generating a new image from scratch, the process remains sequential.
  - We must compute the first pixel, feed it back to compute the second, and so on.
  - Thus, while training is fast, generation is still slow.

> **Key Insight**: Both PixelRNN and PixelCNN provide an explicit, tractable density, meaning we can precisely calculate the likelihood of an image. This gives a highly interpretable evaluation metric. However, their sequential generation mechanism fundamentally limits their speed for synthesizing new data.

## Variational Autoencoders (VAE)

- VAEs take a different approach from PixelCNNs by defining an intractable density function involving a latent variable $z$.

- The model assumes complex data $x$ is generated from an underlying, unobserved representation $z$.

- The probability distribution is an integral over all possible values of $z$:

  $$p_\theta(x) = \int p_\theta(z) p_\theta(x|z) dz$$

- Since this integral is impossible to compute directly, VAEs use a probabilistic formulation to optimize a lower bound on the likelihood.

### Background: Autoencoders

- An autoencoder is an unsupervised neural network designed to learn a lower-dimensional feature representation (latent space) from unlabeled data.

- The architecture has two parts:
  1. **Encoder**: Maps input $x$ to a latent feature vector $z$. The lower dimensionality forces compression, retaining only critical factors of variation.
  2. **Decoder**: Attempts to reconstruct the original input from $z$, producing output $\hat{x}$.

- The network is trained end-to-end to minimize reconstruction error (e.g., L2 loss):
  
  $$L = \|x - \hat{x}\|^2$$

- **Applications**:
  - The trained encoder can be used as a powerful feature extractor for downstream supervised tasks.
  - Allows leveraging structure learned from massive unlabeled datasets for small labeled datasets.

- **Limitations**:
  - Vanilla autoencoders cannot generate entirely new images.
  - The latent space is not regularized or structured. Sampling a random point yields garbage because the space between training points is undefined.

### Variational Autoencoders

- VAEs put a probabilistic spin on traditional autoencoders, structuring the latent space to allow sampling for novel data generation.

- **The key assumption**:
  - Training data is generated from an underlying set of unobserved latent variables $z$.
  - For faces, $z$ might encode attributes like pose, lighting, or smile.

- **The generative process**:
  1. Sample a latent vector $z$ from a true prior distribution $p_{\theta^*}(z)$.
     - Typically, a standard multivariate Gaussian $\mathcal{N}(0, I)$ is chosen, ensuring independent, roughly centered variables.
  2. Sample the image $x$ from the conditional distribution $p_{\theta^*}(x|z)$.
     - Modeled using a powerful neural network (the decoder) due to the non-linear mapping.

- **Goal**: Estimate true parameters $\theta^*$ to maximize the likelihood of the training data.

### The ELBO Derivation

- To maximize data likelihood, we want to maximize $p_\theta(x)$:
  
  $$p_\theta(x) = \int p_\theta(z) p_\theta(x|z) dz$$

- This is intractable because computing $p_\theta(x|z)$ for all possible $z$ is impossible. Using Bayes' rule for the posterior $p_\theta(z|x)$ requires the intractable evidence term $p_\theta(x)$.

- **The VAE Solution**:
  - Introduce an encoder network $q_\phi(z|x)$, parameterized by $\phi$, to approximate the true posterior $p_\theta(z|x)$.
  - The encoder takes image $x$ and outputs the parameters of a distribution over the latent space: a mean vector $\mu_{z|x}$ and a diagonal covariance matrix $\Sigma_{z|x}$.

- **Deriving the Objective**:
  - Express log-likelihood as an expectation over our approximate posterior:
    $$\log p_\theta(x) = \mathbf{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x)]$$
  - Multiply the term inside by $\frac{p_\theta(z|x)}{p_\theta(z|x)}$ and apply Bayes' rule:
    $$\log p_\theta(x) = \mathbf{E}_z \left[ \log \frac{p_\theta(x|z) p_\theta(z)}{p_\theta(z|x)} \right]$$
  - Multiply by $q_\phi(z|x)$ in numerator and denominator:
    $$\log p_\theta(x) = \mathbf{E}_z \left[ \log \frac{p_\theta(x|z) p_\theta(z) q_\phi(z|x)}{p_\theta(z|x) q_\phi(z|x)} \right]$$
  - Split into three terms using log properties:
    $$\log p_\theta(x) = \mathbf{E}_z [\log p_\theta(x|z)] - \mathbf{E}_z \left[ \log \frac{q_\phi(z|x)}{p_\theta(z)} \right] + \mathbf{E}_z \left[ \log \frac{q_\phi(z|x)}{p_\theta(z|x)} \right]$$
  - Recognize the Kullback-Leibler (KL) divergences:
    $$\log p_\theta(x) = \mathbf{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p_\theta(z)) + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

- Since $D_{KL} \geq 0$, we can drop the final term to establish the **Evidence Lower Bound (ELBO)**:
  
  $$\log p_\theta(x) \geq \mathcal{L}(x, \theta, \phi) = \mathbf{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p_\theta(z))$$

- We train the VAE to maximize this lower bound $\mathcal{L}(x, \theta, \phi)$ with respect to $\phi$ and $\theta$.

- **Interpretation**:
  - $\mathbf{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)]$: Acts as a reconstruction loss, encouraging the decoder to accurately reconstruct $x$ from $z$.
  - $- D_{KL}(q_\phi(z|x) \| p_\theta(z))$: Acts as a regularizer, forcing the approximate posterior to be close to the simple prior (Gaussian). This ensures a smooth, continuous latent space suitable for sampling.

### VAE Training

- Training involves computing the ELBO for a minibatch of data.

- **Forward Pass**:
  1. Input image $x$ passes through encoder $q_\phi(z|x)$, yielding $\mu_{z|x}$ and $\Sigma_{z|x}$.
  2. Sample $z \sim \mathcal{N}(\mu_{z|x}, \Sigma_{z|x})$.

- **The Reparameterization Trick**:
  - We cannot backpropagate through a random sampling node.
  - Instead, we sample a noise vector $\epsilon \sim \mathcal{N}(0, I)$ and deterministically compute $z$:
    
    $$z = \mu_{z|x} + \Sigma_{z|x}^{1/2} \odot \epsilon$$
    
  - This allows smooth backpropagation through the mean and variance pathways.

- **Computing Loss**:
  - Sampled $z$ goes through decoder $p_\theta(x|z)$ to output parameters of the distribution over the image (e.g., mean pixel values $\mu_{x|z}$).
  - Total loss is the sum of reconstruction error and analytical KL divergence.

### Generating with VAEs

- Once trained, the encoder network is discarded.

- **Generation**:
  1. Sample a random noise vector $z$ from the prior $z \sim \mathcal{N}(0, I)$.
  2. Pass $z$ through the decoder $p_\theta(x|z)$ to produce a realistic image.

- **Properties of the Latent Space**:
  - The KL divergence regularizer forces the latent space to conform to an independent Gaussian prior, making it smooth and structured.
  - Interpolating between two points in latent space yields smoothly transitioning images.
  - The diagonal prior encourages independent latent variables, leading to interpretable factors of variation (e.g., one dimension alters head pose, another alters smile).

- **Advantages & Disadvantages**:
  - **Pros**: Mathematically principled approach optimizing a clear lower bound; natural inference via the encoder.
  - **Cons**: Because they optimize a lower bound rather than explicitly trying to fool a discriminator, generated samples tend to be blurry and lack high-frequency details.
