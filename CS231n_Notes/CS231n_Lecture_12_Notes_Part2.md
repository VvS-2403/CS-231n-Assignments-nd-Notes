# CS231n Lecture 12: Visualizing and Understanding (Part 2)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
1. [DeepDream: Amplifying Existing Features](#deepdream-amplifying-existing-features)
2. [Feature Inversion](#feature-inversion)
3. [Texture Synthesis: From Classical to Neural](#texture-synthesis-from-classical-to-neural)
4. [Neural Style Transfer](#neural-style-transfer)
5. [Fast Style Transfer and Instance Normalization](#fast-style-transfer-and-instance-normalization)
6. [One Network, Many Styles](#one-network-many-styles)
7. [Summary & Key Takeaways](#summary--key-takeaways)

---

## DeepDream: Amplifying Existing Features

- **Concept:** Unlike Gradient Ascent which synthesizes from noise, DeepDream takes an existing image and **amplifies whatever features the network already sees in it**, across an entire layer.
  - Originally released by Google, it became viral for surreal, psychedelic art.

- **Mathematical formulation:**
  - Maximize the L2 norm of the activations of an entire layer:
    $$I^* = \arg\max_I \sum_i f_i(I)^2$$

- **The algorithm:**
  1. Take a real image and do a forward pass up to a chosen intermediate layer (e.g., `inception_4c`).
  2. Set the gradient at that layer exactly equal to the activation itself: `gradient = activation` (this is the core amplification step).
  3. Backpropagate this synthetic gradient down to the image pixels.
  4. Update the image by making a step in the direction of this gradient.

- **The "Jitter" trick:**
  - To produce coherent, fractal-like patterns, randomly shift (roll) the image by a few pixels before the forward pass, and unshift after the gradient update.
  - This prevents getting stuck on high-frequency artifacts.

- **Code example of a DeepDream step:**
  ```python
  import numpy as np

  def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
      src = net.blobs['data'] # Access the input image tensor
      
      # Jitter image to encourage spatial smoothness and prevent artifacting
      ox, oy = np.random.randint(-jitter, jitter+1, 2)
      src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)
      
      # Forward pass up to the target layer
      net.forward(end=end)
      
      # Set the gradient equal to the activation itself!
      # This is the core "amplification" step.
      net.backward(start=end)
      g = src.diff[0] # Get the computed gradient on the image
      
      # Normalize gradients to ensure stable updates
      src.data[:] += step_size / np.abs(g).mean() * g
      
      # Unjitter image back to original position
      src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)
      
      # Clip pixel values to keep them in valid image range
      if clip:
          bias = net.transformer.mean['data']
          src.data[:] = np.clip(src.data, -bias, 255-bias)
  ```

- **Results based on layer depth:**
  - **Shallow layers:** Amplifies low-level edges/textures, resulting in abstract, impressionist swirling.
  - **Deep layers:** Amplifies high-level semantic objects. Since networks are heavily trained on ImageNet (lots of dogs), it hallucinates dogs, turning clouds into dog-slugs or pagodas.
  - Proves neural networks act as databases of hierarchical visual patterns, imposing their worldview on data.

---

## Feature Inversion

- **The Premise:** Can you reconstruct an original image given only its feature vector from an intermediate layer of a CNN?

- **Optimization Problem:**
  - Let $\Phi_0$ be the target feature vector at a specific layer.
  - Find a new image $\mathbf{x}$ whose feature vector $\Phi(\mathbf{x})$ matches $\Phi_0$.
  - Minimize L2 distance, regularized by an image prior (Total Variation regularizer to ensure smoothness):
    $$\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathbb{R}^{H \times W \times C}} \|\Phi(\mathbf{x}) - \Phi_0\|^2 + \lambda \mathcal{R}_{TV}(\mathbf{x})$$

- **Results of inverting different layers (e.g., in VGG-16):**
  - **Shallow layer (e.g., `relu2_2`):** Reconstructed image is nearly indistinguishable from original. Pixel information is perfectly preserved.
  - **Deep layer (e.g., `relu5_1`):** Reconstructed image looks profoundly different. Exact colors and fine details are destroyed, but overall semantic structure (silhouette, layout) remains intact.

- **Fundamental theorem proved:** As we go deeper into a neural network, representations discard trivial pixel-level variations and increasingly encode pure, abstract semantic content.

---

## Texture Synthesis: From Classical to Neural

- **Goal:** Take a small sample patch of texture and generate a much larger image of the exact same texture seamlessly.

- **Classical approach (Nearest Neighbor):**
  - Generates new image pixel by pixel.
  - Looks at the local neighborhood of already-generated pixels, searches original patch for a match, and copies the central pixel.
  - Fails on complex textures requiring broader spatial understanding.

### Neural Texture Synthesis: Gram Matrix

- **Moving to neural feature space:**
  - Each layer produces an activation tensor of $C \times H \times W$.
  - Viewed as an $H \times W$ grid of $C$-dimensional vectors representing visual features at spatial locations.

- **Capturing texture with Gram Matrices:**
  - Texture depends on which features co-occur (e.g., "brown" and "vertical edge" activating together for wood).
  - The Gram Matrix captures this by taking the outer product of two $C$-dimensional vectors and averaging over spatial locations.
  - Formula for Gram Matrix at layer $l$:
    $$G^l_{ij} = \sum_k F^l_{ik} F^l_{jk}$$
  - $F^l$ is reshaped feature matrix size $C_l \times (H_l W_l)$.
  - In code: $G = F F^T$.
  - It completely discards spatial info and retains the statistical distribution of features (the definition of texture).

- **Neural Texture Synthesis Algorithm:**
  1. Pretrain a CNN (VGG-19) on ImageNet.
  2. Run input texture through VGG, record activation maps at various layers.
  3. Compute Gram matrices (target texture representation).
  4. Initialize generated image from random noise.
  5. Pass generated image through VGG, compute *its* Gram matrices.
  6. Calculate loss (weighted sum of L2 distances between generated and target Gram matrices):
     $$E_l = \frac{1}{4N_l^2M_l^2} \sum_{i,j} (G^l_{ij} - \hat{G}^l_{ij})^2$$
     $$\mathcal{L}(\vec{x}, \hat{\vec{x}}) = \sum_{l=0}^L w_l E_l$$
  7. Backpropagate loss to get gradient on image pixels.
  8. Make a gradient step.
  9. Repeat until Gram matrices match perfectly.
  - Produces infinitely large, seamless textures of complex structures.

---

## Neural Style Transfer

- **The Concept:** Combines Feature Inversion (content) with Neural Texture Synthesis (style). Takes a **Content Image** (photograph) and a **Style Image** (painting), outputting the photograph painted in the style of the painting.

- **Joint Optimization:** Minimizes two losses simultaneously:
  1. **Content Loss:** L2 distance between feature maps of content image and generated image at a deep layer (e.g., `relu4_2`). Preserves semantic layout.
  2. **Style Loss:** L2 distance between Gram matrices of style image and generated image across multiple layers. Preserves texture and color.

- **Total Loss:**
  $$\mathcal{L}_{total} = \alpha \mathcal{L}_{content} + \beta \mathcal{L}_{style}$$
  - $\alpha$ and $\beta$ hyperparameters control priority (structural integrity vs. aggressive stylization).

- **Flexibility:**
  - Resizing style image changes the scale of transferred texture (e.g., brushstroke size).
  - Can blend styles by taking the weighted average of Gram matrices from two different style images.

---

## Fast Style Transfer and Instance Normalization

- **The fatal flaw of Neural Style Transfer:** It is agonizingly slow, requiring solving a complex optimization loop for every single image (takes minutes on a GPU).

- **Fast Style Transfer:**
  - Shifts computational burden from inference to training.
  - Trains a separate **Feedforward Network** to do style transfer in a single pass.
  - **Training process:**
    1. Pick a specific style.
    2. Take a massive dataset of content images.
    3. Pass a content image through the untrained Feedforward Network to get an output.
    4. Pass output, original content, and style image into a frozen, pretrained VGG network ("Loss Network").
    5. VGG computes Content and Style Loss.
    6. Backpropagate loss to update the *Feedforward Network* weights.
  - At inference time, stylizing an image takes milliseconds (real-time video).

- **Instance Normalization:**
  - Discovered during development of Fast Style Transfer.
  - Replaces Batch Normalization.
  - Normalizes activations independently for *each single image* across its spatial dimensions.
  - Prevents the styles of different images in a batch from polluting one another, drastically improving visual quality.

---

## One Network, Many Styles

- **The limitation of Fast Style Transfer:** Requires a separate Feedforward Network for every style (memory prohibitive for many styles).

- **Conditional Instance Normalization:**
  - Allows a *single* network to learn many styles.
  - Standard normalization applies learned scale $\gamma$ and shift $\beta$:
    $$z = \gamma x_{norm} + \beta$$
  - Conditional IN learns a massive lookup table containing separate $\gamma_s$ and $\beta_s$ for *each specific style* $s$:
    $$z = \gamma_s x_{norm} + \beta_s$$
  - During training, randomly select a style, look up its parameters, apply them, and update both shared conv weights and style-specific params.
  - Conv filters learn general drawing features; IN params act as "switches" for specific styles.

- **Real-time Style Blending:**
  - Can blend styles on the fly without retraining by taking a convex combination of normalization parameters:
    $$\gamma_{blend} = 0.5 \gamma_{vangogh} + 0.5 \gamma_{picasso}$$
    $$\beta_{blend} = 0.5 \beta_{vangogh} + 0.5 \beta_{picasso}$$

---

## Summary & Key Takeaways

- **Activations & Features:**
  - Visualizing first-layer filters, using nearest-neighbor in FC7 space, and t-SNE clustering reveals how networks learn semantics.
  - Activation maps and maximally activating patches show neurons act as specialized feature detectors.
- **Gradients:**
  - Backpropagation can generate Saliency Maps to highlight important classification pixels (useful for segmentation).
  - Guided Backpropagation produces interpretable visualizations of intermediate neurons by masking negative gradients.
- **Optimization & Generation:**
  - Gradient Ascent on pixels can synthesize perfect class representations.
  - DeepDream hallucinates features.
  - Feature Inversion reveals what information is preserved at different depths.
  - Adversarial Examples show CNN brittleness to imperceptible changes.
- **Style Transfer:**
  - Gram matrices synthesize texture.
  - Jointly matching Gram matrices (style) and deep features (content) achieves Neural Style Transfer.
- **Real-time Deployment:**
  - Fast Style Transfer approximates optimization in a single pass via a Feedforward Network.
  - Conditional Instance Normalization lets one network store and interpolate dozens of styles on the fly.
