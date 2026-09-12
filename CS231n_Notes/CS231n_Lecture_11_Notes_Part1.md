 # CS231n Lecture 11: Detection and Segmentation (Part 1)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Computer Vision Tasks Overview

- While image classification has historically been the primary focus of early deep learning breakthroughs, the real world demands much more nuanced understanding of visual scenes.
- Image classification simply answers the question, "What is the main object in this image?" by assigning a single label to the entire image.
- However, real images contain complex scenes with many objects, overlapping elements, and diverse backgrounds.
- To handle this, computer vision has evolved several distinct tasks with increasing levels of granularity.

### Semantic Segmentation
- The goal here is to label every single pixel in the image with a category label (such as "sky," "car," "person," or "tree").
- It is a dense prediction task.
- Importantly, semantic segmentation does not differentiate between separate instances of the same object class.
- If there are two cars next to each other, the pixels for both cars will simply be labeled as "car," forming one large blob of car pixels without separating the two vehicles.

### Classification + Localization
- In this task, we assume there is exactly one main object in the image.
- The goal is twofold: 
  - First, we must classify the object (what is it?).
  - Second, we must localize it by predicting a bounding box around it (where is it?).
- The bounding box is typically represented by four numbers (e.g., $x, y, \text{width}, \text{height}$).

### Object Detection
- This task removes the assumption that there is only one object.
- In object detection, an image might contain zero, one, or dozens of objects of various classes.
- The algorithm must identify all objects, classify each one, and draw a 2D bounding box around every instance.
- This is fundamentally more challenging because the number of outputs per image is variable.

### Instance Segmentation
- This is the most complex of the core tasks, combining object detection and semantic segmentation.
- The goal is to detect all distinct objects in an image and, rather than just drawing a bounding box, predicting the exact pixels that belong to each individual object.
- Unlike semantic segmentation, it separates instances: car 1 and car 2 will be identified as distinct objects with their own individual masks.

---

## Semantic Segmentation

- The task of semantic segmentation requires us to make a categorical prediction for every pixel in the input image.
- Let's look at three progressive ideas that the field explored to solve this problem, culminating in the architectures used today.

### Idea 1: Sliding Window
- The most naive approach to dense pixel labeling is the sliding window method.
- Imagine taking a small window or patch of the image, centering it on a specific pixel.
- We extract this patch, pass it through a standard Convolutional Neural Network (CNN), and use the CNN to predict a single class label for the center pixel.
- We then slide this window across every single pixel in the image, repeating the process.

- **Why it's inefficient:**
  - This method is catastrophically inefficient.
  - Consider a small image of $256 \times 256$ pixels. This requires $65,536$ separate forward passes through the entire CNN.
  - Worse, overlapping patches share almost all of the same pixels, meaning the CNN is doing redundant computation.
  - It extracts the exact same low-level features (like edges and colors) over and over again for the same regions of the image.
  - The computational cost makes this completely infeasible for practical use.

### Idea 2: Fully Convolutional Network (FCN)
- To solve the redundancy of the sliding window, researchers realized they could design a network consisting entirely of convolutional layers, without any fully connected layers at the end.
- This allows the network to process the whole image in a single forward pass and output a spatial map of predictions.
- In an FCN, the input is an image of size $3 \times H \times W$.
- The network applies a series of convolutions, preserving the spatial dimensions (using appropriate zero-padding).
- The intermediate feature maps might have dimensions $D \times H \times W$.
- The final convolutional layer predicts a set of scores for each class at every pixel, resulting in an output tensor of size $C \times H \times W$, where $C$ is the number of classes.
- We can then take the argmax across the channel dimension to get the final $H \times W$ prediction map.

- **Why it's computationally expensive:**
  - While this solves the redundant computation problem, it introduces massive memory and compute issues.
  - Processing an image at its original, high resolution through many deep convolutional layers requires an enormous amount of memory to store the activations (which are needed for backpropagation).
  - Furthermore, keeping the spatial resolution large means each convolution operation is performing a huge number of floating-point operations.
  - Finally, keeping the resolution identical means the receptive field of the neurons grows very slowly, making it hard for the network to understand global context.

### Idea 3: Downsampling and Upsampling
- The modern solution to semantic segmentation is an encoder-decoder architecture.
- The first half of the network (the encoder) looks like a standard classification CNN.
- It uses pooling layers and strided convolutions to progressively downsample the spatial resolution of the feature maps, while increasing the depth (number of channels).
- For instance, an input of $3 \times H \times W$ might be reduced to $D \times (H/32) \times (W/32)$.
- This downsampling solves the memory and compute problems, allowing the network to process the image efficiently and develop large receptive fields that capture deep semantic context.
- However, the output is now way too small; we need predictions at the original $H \times W$ resolution.
- This is where the second half of the network (the decoder) comes in.
- It uses **upsampling** layers to progressively increase the spatial dimensions of the feature map, restoring it back to $H \times W$.
- The network is trained end-to-end to predict the pixel labels.

---

## Upsampling Methods

- Since downsampling throws away spatial information, upsampling must intelligently expand small feature maps back into larger ones.
- There are several ways to perform upsampling inside a neural network.

### Nearest Neighbor Unpooling
- The simplest approach.
- To upsample a $1 \times 1$ region to a $2 \times 2$ region, we simply duplicate the single value into all four positions of the $2 \times 2$ output.
- It's fast and requires no parameters, but produces very blocky, unrefined feature maps.

### Bed of Nails Unpooling
- In this method, the single input value is placed in the top-left corner of the $2 \times 2$ output region, and the other three positions are filled with zeros.
- It's called "bed of nails" because a flat feature map upsampled this way looks like a flat surface with sharp spikes (the values) surrounded by empty space (zeros).

### Max Unpooling
- This is a clever trick specifically paired with Max Pooling layers from the encoder.
- When a Max Pooling layer downsamples a region, it throws away all values except the maximum.
- However, it can "remember" the spatial index (the switch position) of that maximum value.
- During the decoder phase, the Max Unpooling layer takes the single value to be upsampled and places it precisely in the spatial position that was recorded by the corresponding Max Pooling layer in the encoder.
- The rest of the $2 \times 2$ region is filled with zeros.
- This helps preserve precise spatial boundary information that would otherwise be permanently lost during pooling.

### Learnable Upsampling: Transposed Convolution
- The methods above are fixed functions.
- What if we want the network to *learn* how to upsample optimally?
- This is achieved using **Transposed Convolution** (sometimes poorly named "deconvolution" or "fractionally strided convolution").
- In a normal convolution, we take a filter, compute the dot product with a local region of the input, and output a single scalar value. Stride greater than 1 causes downsampling.
- In a transposed convolution, the mechanics are flipped. We take a single scalar value from the input and multiply it by the entire filter (the weights).
- This scaled filter is then "stamped" onto the output feature map.
  - Input value: $x$ (a scalar)
  - Filter: $\mathbf{W}$ (e.g., a $3 \times 3$ matrix)
  - Output stamp: $x \mathbf{W}$
- If we use a stride of 2 in a transposed convolution, we move our focus by 1 pixel in the input space, but we move the "stamp" by 2 pixels in the output space.
- This effectively upsamples the image.
- When the stamped regions in the output overlap, the values are simply **summed** together.

- **Matrix Multiplication View:**
  - A normal convolution can be expressed as a matrix multiplication $\mathbf{y} = \mathbf{X}\mathbf{w}$, where $\mathbf{X}$ is a specially constructed matrix of the input (via im2col).
  - The backward pass (gradient) of this operation involves multiplying by the transpose, $\mathbf{X}^T$.
  - A transposed convolution gets its name because its forward pass mathematically mimics the backward pass of a normal convolution; it applies the filter via $\mathbf{y} = \mathbf{X}^T\mathbf{w}$.

---

## Skip Connections in Segmentation

- While the encoder-decoder architecture with transposed convolutions works well, it suffers from a fundamental tension.
- The deepest layers of the encoder (the bottleneck) contain rich, highly abstract semantic information ("this is a cat").
- However, they have lost almost all precise spatial information ("exactly where are the cat's fur boundaries?").
- To resolve this, architectures like **U-Net** introduce **skip connections**.
- A skip connection takes high-resolution, low-semantic feature maps from the early layers of the encoder and directly concatenates them (or adds them) to the upsampled feature maps in the decoder.
- This provides the decoder with the best of both worlds:
  - It gets high-level context from the deep network path.
  - It gets crisp spatial boundary details directly from the early layers.
- This allows it to produce highly accurate, pixel-perfect segmentation masks.

---

## Classification + Localization

- When moving from pure classification to classification plus localization, we assume there is exactly one main object in the image.
- We treat localization as a **regression problem**.
- The architecture typically starts with a standard CNN backbone (like ResNet or VGG) that processes the image and extracts a global feature vector.
- From this feature vector, the network splits into two parallel fully connected (FC) heads:

  1. **Classification Head:**
     - An FC layer that predicts the class scores.
     - It outputs a vector of size $C$ (number of classes).
     - This is passed through a softmax function to compute a **Softmax Loss** (Cross-Entropy).

  2. **Regression Head:**
     - An FC layer that predicts the bounding box coordinates.
     - It outputs exactly 4 numbers: $(x, y, w, h)$, representing the center coordinates, width, and height of the box.
     - We compute an **L2 Loss** (or Smooth L1 loss) between these predicted coordinates and the ground truth coordinates $(x', y', w', h')$.

- These two losses are summed together to form a **Multitask Loss**:
  $$ L = L_{classification} + \lambda L_{regression} $$
  where $\lambda$ is a hyperparameter balancing the two objectives.

> **Key Insight:** Training such networks from scratch is difficult. In practice, the CNN backbone is almost always pre-trained on ImageNet (a massive classification dataset). We use Transfer Learning by taking the pre-trained weights, adding the new regression head, and fine-tuning the entire system for the localization task.
