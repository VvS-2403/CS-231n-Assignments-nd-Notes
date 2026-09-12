# CS231n Lecture 12: Visualizing and Understanding (Part 1)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
1. [Motivation: What is going on inside ConvNets?](#motivation-what-is-going-on-inside-convnets)
2. [First Layer: Visualizing Filters](#first-layer-visualizing-filters)
3. [Last Layer Analysis: Features and Dimensionality Reduction](#last-layer-analysis-features-and-dimensionality-reduction)
4. [Visualizing Activations and Maximally Activating Patches](#visualizing-activations-and-maximally-activating-patches)
5. [Which Pixels Matter: Saliency vs. Occlusion](#which-pixels-matter-saliency-vs-occlusion)
6. [Saliency via Backpropagation and Segmentation](#saliency-via-backpropagation-and-segmentation)
7. [Intermediate Features via Guided Backpropagation](#intermediate-features-via-guided-backpropagation)
8. [Visualizing CNN Features: Gradient Ascent](#visualizing-cnn-features-gradient-ascent)
9. [Fooling Images and Adversarial Examples](#fooling-images-and-adversarial-examples)

---

## Motivation: What is going on inside ConvNets?

- **The "Black Box" problem:** For much of deep learning's history, CNNs have been criticized as opaque.
  - We feed an image into a complex architecture with millions of parameters and dozens of layers.
  - It outputs a probability distribution over object categories.
  - While accurate, the internal mechanics often seem magical and incomprehensible.
  - We want to know exactly what happens inside a CNN between raw pixels and the final decision.

- **Why understand intermediate features?**
  - It is not just intellectual curiosity.
  - It is crucial for debugging, building trust in AI systems, and designing better architectures.
  - Consider standard architectures like AlexNet or VGG-16: an input image of 3 × 224 × 224 pixels passes through convolutional, pooling, and fully-connected layers, outputting 1000 class scores.
  - What are the intermediate features looking for? How do pixels turn into concepts like "dog" or "car"?

> **Key Insight**: To unpack the black box of deep learning, we must develop techniques to visualize and interpret the raw weights, the intermediate activations, and the gradients of the network, tracing how pixel-level data is hierarchically abstracted into high-level semantics.

---

## First Layer: Visualizing Filters

- **Direct observation:** The simplest way to investigate a CNN is to look directly at the learned weights.

- **First-layer filters:**
  - They interact directly with raw image pixels.
  - This makes them highly interpretable.
  - In AlexNet, the first layer has 64 filters of size 3 × 11 × 11.
  - Since they operate on 3 RGB channels, we can visualize each filter as an 11 × 11 RGB image.

- **Universal features:**
  - When plotting these filters, a striking pattern emerges across all architectures (AlexNet, ResNet, DenseNet).
  - The first layer always seems to learn the exact same types of features.

- **Two primary types of filters:**
  - **Oriented edges:** Black-and-white bars at various angles (acting like Gabor filters for edges).
  - **Opposing color blobs:** Bright color centers fading into different backgrounds, detecting color transitions.
  - This matches biological visual processing (the V1 cortex in the human brain relies on oriented edge detectors).

- **The limit of direct visualization:**
  - This technique loses utility deeper in the network.
  - Higher-layer filters (e.g., Layer 2 with 20 × 16 × 7 × 7 weights) are no longer simple RGB images.
  - Slicing these weights tells us little, because Layer 2 looks at Layer 1's activation map, not raw pixels.
  - Therefore, visualizing raw weights is primarily for the very first layer.

---

## Last Layer Analysis: Features and Dimensionality Reduction

- **Looking at the final layer:** If the first layer gives low-level edges, what does the final layer (just before the classifier, like FC7) give us?

- **The holistic summary:**
  - By the time data reaches FC7, all spatial dimensions have collapsed.
  - The image is represented as a single, dense, 4096-dimensional feature vector.
  - To understand this space, we run a massive dataset through the network and analyze the collected FC7 vectors.

- **Nearest neighbors comparison:**
  - **Pixel space:** Finding L2 nearest neighbors in raw pixels yields nonsensical semantic results (e.g., a dark dog matches a dark car due to brightness and color). Pixel space is highly sensitive to background, lighting, and translation.
  - **Feature space (FC7):** Finding L2 nearest neighbors in FC7 space yields semantically similar results. A query elephant image retrieves other elephants, regardless of background, pose, or camera angle.
  - The network has learned to discard irrelevant variations and encode the core semantic identity.

### Dimensionality Reduction: PCA and t-SNE

- **Visualizing 4096 dimensions:** It's impossible for humans, so we squash the space to 2 or 3 dimensions for a scatterplot.

- **Principal Component Analysis (PCA):**
  - A linear algorithm that finds directions of maximum variance and projects vectors onto the top 2 components.
  - Efficient but often fails to capture the complex, non-linear structure of neural network feature spaces.

- **t-Distributed Stochastic Neighbor Embedding (t-SNE):**
  - A non-linear algorithm designed to preserve local structure.
  - Points close in 4096-D space remain close in 2D.
  - When applied to FC7 features, it naturally forms distinct, isolated clusters of semantically similar images (without knowing the class labels).
  - Dogs cluster with dogs, airplanes with airplanes, proving the CNN learned a structured semantic landscape.

---

## Visualizing Activations and Maximally Activating Patches

- **Intermediate layers:** How do we explore layers between the first and last?

- **Visualizing activation maps:**
  - In a forward pass, a layer like `conv5` might output a feature map of 128 × 13 × 13.
  - We can visualize these 128 channels as a grid of grayscale images (brighter pixels = higher activation).
  - We notice specific channels act as specialized detectors (e.g., channel 17 might light up intensely over a person's face).
  - The network spontaneously learns these detectors without explicit bounding box supervision.

- **Maximally Activating Patches:**
  - Used to rigorously confirm what a specific neuron (channel) is looking for.
  - **Procedure:**
    1. Pick a layer and channel.
    2. Run a massive dataset through the network, recording the activation value for every image.
    3. Sort the images by activation value.
    4. Extract the specific spatial patches that caused the highest activations.
  - The resulting grid of patches reveals the channel's preferred stimulus (e.g., faces, text, dog snouts).
  - Since it's a deeper layer, the receptive field is large, allowing it to respond to complex structural patterns rather than just simple edges.

---

## Which Pixels Matter: Saliency vs. Occlusion

- **Goal:** For a *particular* image, which specific pixels caused the network to classify it the way it did?

- **Occlusion Experiments:**
  - A brute-force approach.
  - Systematically slide a gray square (mask) across the input image.
  - For every position, run the occluded image through the network and record the predicted probability of the true class.
  - Sliding over background won't change the probability much.
  - Sliding over crucial features (like an elephant's trunk) causes the probability to plummet.
  - We generate a heat map mapping predicted probability to occlusion position, highlighting critical regions.
  - **Drawback:** Computationally expensive, requiring thousands of forward passes for a single image.

---

## Saliency via Backpropagation and Segmentation

- **Saliency Maps:** A more computationally efficient approach using backpropagation.

- **The process:**
  1. Perform a forward pass with the input image to compute unnormalized class scores (logits).
  2. Isolate the score for the correct class (e.g., $S_c$).
  3. Perform a backward pass, computing the gradient of the class score with respect to the *input image pixels*, not the weights:
     $$ \nabla_I S_c $$
  - This gradient tells us how much the class score changes if we alter each pixel slightly.
  - Large gradient magnitudes indicate high influence on the class score.

- **Creating the 2D map:**
  - Take the absolute value of the gradients and the maximum across the three color channels for each pixel.
  - The result is a ghostly silhouette of the object (bright pixels outline the object).

- **Segmentation without Supervision:**
  - Saliency maps effectively localize objects.
  - By thresholding a saliency map and feeding it to a graph-cut algorithm (like GrabCut), we can automatically extract a highly accurate foreground mask.
  - The network learns to segment objects entirely as a byproduct of learning to classify them!

---

## Intermediate Features via Guided Backpropagation

- **Understanding intermediate neurons:** We can compute the gradient of any single intermediate neuron's activation with respect to the input image.
  - Do a forward pass, set the target neuron's gradient to 1 and all others to 0.
  - Backpropagate to the image pixels to see which pixels excite that neuron.

- **The problem with standard backprop:**
  - Backpropagation through ReLUs often yields noisy, visually unintelligible results because negative gradients (pixels that *decrease* activation) muddy the visualization.

- **Guided Backpropagation:**
  - Forcefully prevents negative gradients from flowing backward through ReLUs.
  - Standard ReLU backprop only passes gradient if forward input was positive:
    ```python
    dx = (x > 0) * dout
    ```
  - Guided Backprop adds a condition: only pass the gradient if the incoming gradient itself is also positive. Zero out negative gradients:
    ```python
    dx = (x > 0) * (dout > 0) * dout
    ```

- **Result:**
  - Explicitly masks out negative gradients, keeping only pixel features that strictly *increase* the target neuron's activation.
  - Produces crisp, high-frequency edge maps highlighting exact structural features (like outlines of eyes and noses for a face detector).

---

## Visualizing CNN Features: Gradient Ascent

- **Synthesizing ideal images:** What if we want to know what a neuron looks for independent of any input image? We use Gradient Ascent.

- **The optimization problem:**
  $$I^* = \arg\max_I f(I) + \lambda R(I)$$
  - $I$: the image to synthesize.
  - $f(I)$: the activation value to maximize.
  - $R(I)$: a regularization term.
  - $\lambda$: hyperparameter for regularization strength.

- **The algorithm:**
  - Works like training a network, but in reverse.
  - **Freeze all network weights** and update **input image pixels**.
  1. Initialize $I$ to random noise.
  2. Forward pass through frozen network to compute $f(I)$.
  3. Backpropagate to compute gradient: $\nabla_I f(I)$.
  4. Make a small step: $I = I + \alpha \nabla_I f(I)$.
  5. Apply regularization to $I$.
  6. Repeat steps 2-5.

> **Key Insight**: While Guided Backprop finds the part of an existing image that a neuron responds to, Gradient Ascent synthesizes a completely new, artificial image that represents the platonic ideal of what the neuron wants to see.

### The Crucial Role of Regularization

- **Why regularization is needed:**
  - Without $R(I)$, the image mathematically maximizes the neuron but looks like high-frequency static noise to humans.
  - Networks are non-linear; it's easy to find unnatural pixel configurations that "hack" high activations.

- **Regularization techniques:**
  - L2 norm penalty: $R(I) = -\|I\|_2^2$ (keeps values from blowing up).
  - Cocktail of heuristic regularizers during each step:
    1. Apply **Gaussian blur** (penalizes high-frequency noise, encourages smoothness).
    2. Apply **pixel clipping** (forces small values to 0).
    3. Apply **gradient clipping** (ignores very small gradients).

- **Results:**
  - Synthesizes remarkable images (e.g., maximizing "dumbbell" creates a ghostly dumbbell; "dalmatian" creates a pattern of black and white spots).
  - Proves weights intrinsically encode visual properties.
  - Neurons can be multi-faceted (e.g., "grocery store" neuron responds to apples, shelves, facades).
  - Optimizing in a deep latent space (like FC6) and decoding to an image produces more photorealistic results.

---

## Fooling Images and Adversarial Examples

- **Exploiting the optimization:** The machinery of Gradient Ascent leads to a concerning discovery: Adversarial Examples.

- **The process to fool a network:**
  1. Start with a pristine image correctly classified with high confidence (e.g., an elephant).
  2. Arbitrarily pick an unrelated target class (e.g., "koala").
  3. Freeze the network weights.
  4. Perform a forward pass, compute loss for "koala", and backpropagate to get the gradient on image pixels.
  5. Make a tiny update to the pixels maximizing the "koala" score.
  6. Constrain updates so they are infinitesimally small (invisible to a human).

- **The terrifying result:**
  - The network classifies the identical-looking image as a "koala" with 99% confidence.
  - The carefully crafted, invisible noise perfectly aligns with the high-dimensional decision boundary for "koala".

> **Key Insight**: Neural networks are extremely brittle to targeted, adversarial perturbations. We can easily fool neural networks into making completely incorrect classifications with high confidence by making imperceptible changes to the input pixels.
