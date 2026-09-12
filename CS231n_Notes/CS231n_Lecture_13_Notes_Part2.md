# CS231n Lecture 13: Generative Models (Part 2)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Generative Adversarial Networks (GAN)](#generative-adversarial-networks-gan)
  - [The Two-Player Game](#the-two-player-game)
  - [Minimax Objective](#minimax-objective)
  - [The Gradient Saturation Problem](#the-gradient-saturation-problem)
  - [GAN Training Algorithm](#gan-training-algorithm)
  - [DCGAN](#dcgan)
  - [Interpretable Vector Math](#interpretable-vector-math)
  - [GAN Variants](#gan-variants)
- [Summary Comparison](#summary-comparison)

---

## Generative Adversarial Networks (GAN)

- If the goal is to generate the most realistic, high-quality images possible—and we are willing to abandon explicit density modeling—we turn to GANs.

- GANs are implicit density models.
  - They do not mathematically define a probability distribution function.
  - Instead, they frame generation as a game-theoretic competition between two opposing neural networks.

- The solution approach:
  - Take a sample from a simple distribution (like random noise).
  - Learn a complex transformation from that noise space to the true data distribution using a neural network.

### The Two-Player Game

- A GAN consists of two entirely separate neural networks locked in a minimax game:

  1. **The Generator Network ($G$)**:
     - Goal: Synthesize realistic images that pass as genuine data (like a counterfeiter).
     - Input: Random noise vector $z$.
     - Output: Generated image $G(z)$.

  2. **The Discriminator Network ($D$)**:
     - Goal: Act as an expert detective to distinguish real from fake.
     - Input: A mixture of real images $x$ from the dataset and fake images $G(z)$.
     - Output: A probability value indicating how likely the input image is real.

- **Training Dynamics**:
  - As $G$ produces better fakes, $D$ is forced to identify more subtle flaws.
  - As $D$ becomes more perceptive, $G$ is forced to learn deeper statistics of the true data distribution to fool it.
  - At theoretical equilibrium, $G$ produces perfect fakes, and $D$ simply guesses randomly (50% probability).
  - Post-training, $D$ is discarded, and $G$ is used to synthesize novel data.

### Minimax Objective

- The joint training is formalized in a minimax objective function.

- Let $\theta_g$ be generator parameters, and $\theta_d$ be discriminator parameters:

  $$\min_{\theta_g} \max_{\theta_d} \left[ \mathbf{E}_{x \sim p_{data}} [\log D_{\theta_d}(x)] + \mathbf{E}_{z \sim p(z)} [\log(1 - D_{\theta_d}(G_{\theta_g}(z)))] \right]$$

- **Breaking down the terms**:
  - $D_{\theta_d}(x)$: Discriminator's predicted probability that real image $x$ is real.
  - $D_{\theta_d}(G_{\theta_g}(z))$: Discriminator's predicted probability that fake image $G(z)$ is real.

- **Discriminator's Goal ($\theta_d$)**:
  - Wants to **maximize** the expression.
  - Pushes $D(x)$ close to 1 (maximizing the first term).
  - Pushes $D(G(z))$ close to 0 (maximizing the second term to $\log(1 - 0) = 0$).

- **Generator's Goal ($\theta_g$)**:
  - Wants to **minimize** the expression.
  - It controls only the second term.
  - Pushes $D(G(z))$ close to 1, causing the second term to drop toward $-\infty$ (fooling the discriminator).

### The Gradient Saturation Problem

- Alternating gradient descent encounters a practical issue early in training: the gradient saturation problem.

- **The Problem**:
  - The generator wants to minimize $\log(1 - D(G(z)))$.
  - Initially, generated samples are random noise, so $D(G(z))$ is very close to 0.
  - Near $x=0$, the slope of $\log(1 - x)$ is relatively flat.
  - Thus, when the generator is performing worst and needs the strongest gradient signal, the gradient is incredibly small.

- **The Solution (Optimization Trick)**:
  - Change the generator's objective from minimizing the likelihood that the discriminator is correct to **maximizing the likelihood that the discriminator is wrong**:

    $$\max_{\theta_g} \mathbf{E}_{z \sim p(z)} [\log(D_{\theta_d}(G_{\theta_g}(z)))]$$

  - This provides the same optimal fixed point but changes the loss landscape.
  - Now, when $D(G(z))$ is near 0, $\log(x)$ has an extremely steep gradient, providing a strong learning signal precisely when needed.

### GAN Training Algorithm

- Training alternates between optimizing the discriminator and generator.

- **1. Discriminator Update Phase** (for $k$ steps, often $k=1$):
  - Sample $m$ noise vectors $\{z^{(1)}, \dots, z^{(m)}\}$ from prior $p(z)$.
  - Sample $m$ real examples $\{x^{(1)}, \dots, x^{(m)}\}$ from training data.
  - Generate $m$ fake images via a forward pass.
  - Ascend the stochastic gradient for $\theta_d$ to maximize real vs fake distinction:
    
    $$\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m \left[ \log D_{\theta_d}(x^{(i)}) + \log(1 - D_{\theta_d}(G_{\theta_g}(z^{(i)}))) \right]$$

- **2. Generator Update Phase**:
  - Sample a *new* minibatch of $m$ noise vectors (to prevent overfitting to the previous batch's noise).
  - Ascend the stochastic gradient for $\theta_g$ using the improved objective:
    
    $$\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log(D_{\theta_d}(G_{\theta_g}(z^{(i)})))$$

- **Challenges**:
  - Jointly training two models is highly unstable and heavily dependent on hyperparameters.
  - Susceptible to **mode collapse** (the generator produces a very limited variety of images).

### DCGAN

- Deep Convolutional GANs (DCGAN) introduced architectural guidelines that dramatically stabilized training and improved image quality for high-resolution generation.

- **Architecture**:
  - **Generator**: Takes noise $z$ (e.g., 100-D), reshapes to a small spatial feature map, and uses fractionally-strided convolutions to progressively upsample until outputting a 3-channel image.
  - **Discriminator**: A standard CNN that downsamples an image to a single probability score.

- **Five Key Guidelines**:
  1. Replace max-pooling with strided convolutions (discriminator) and fractional-strided convolutions (generator) for learned spatial resizing.
  2. Use Batch Normalization in both models to stabilize learning and ensure healthy gradient flow.
  3. Remove fully connected hidden layers in deep architectures, relying solely on convolutions.
  4. Use ReLU activation in the generator, except for the final output layer (which must use Tanh to squash pixels between -1 and 1).
  5. Use LeakyReLU activation in the discriminator to prevent "dead" neurons and allow gradients to flow for negative outputs.

### Interpretable Vector Math

- A fascinating property of GANs is the structure of the learned latent space $Z$, where semantic concepts organize linearly despite a lack of explicit labels.

- We can perform vector arithmetic in the latent space.
  - Example: (Smiling Woman) - (Neutral Woman) + (Neutral Man) = (Smiling Man).
  - Manipulating vectors can seamlessly add or remove glasses from a face.

- This demonstrates that independent dimensions/directions in the latent space correspond to high-level semantic attributes.

### GAN Variants

- The "GAN Zoo" has exploded as researchers modify objective functions and architectures to improve stability and fidelity.

- **Notable Variants**:
  - **Wasserstein GAN (WGAN)**: Uses Earth Mover's distance for a smooth loss metric that correlates with sample quality, mitigating gradient saturation and mode collapse.
  - **Conditional GANs**: Allow users to control the class of generated images.
  - **CycleGAN**: Translates images between domains (e.g., horses to zebras) using unpaired datasets and cycle-consistency loss.
  - **Pix2pix**: Transforms sketches into photorealistic images, or segmentation maps into scenes.
  - **Text-to-image models**: Synthesize images directly from descriptive captions.

## Summary Comparison

- **PixelRNN and PixelCNN**:
  - Explicit density models with a tractable, exact likelihood.
  - Excellent for evaluation and produce good samples.
  - Sequential generation makes them prohibitively slow for real-time synthesis.

- **Variational Autoencoders (VAEs)**:
  - Explicit approximate density models optimizing a variational lower bound (ELBO).
  - Strong mathematically principled framework and natural inference capabilities.
  - Samples tend to appear blurrier than state-of-the-art models due to optimizing a lower bound.

- **Generative Adversarial Networks (GANs)**:
  - Implicit density models using a two-player adversarial game.
  - Produce the sharpest, most beautiful state-of-the-art samples.
  - Notoriously finicky, trickier to train, and do not natively provide a way to map existing images back to latent vectors.
