# CS231n Lecture 5: Convolutional Neural Networks (Part 1)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [ConvNet Layers](#convnet-layers)
- [Convolutional Layer](#convolutional-layer)
  - [Local Connectivity](#local-connectivity)
  - [Spatial Arrangement](#spatial-arrangement)
  - [Parameter Sharing](#parameter-sharing)
  - [Matrix Multiplication Implementation](#matrix-multiplication-implementation)
  - [Backpropagation](#backpropagation)

---

## Architecture Overview

Regular Neural Nets don't scale well to full images. In CIFAR-10, images are size 32x32x3. A single fully-connected neuron in a first hidden layer would have 32*32*3 = 3072 weights. For an image of size 200x200x3, neurons would have 120,000 weights. This full connectivity is wasteful and leads to overfitting.

**[KEY CONCEPT] 3D volumes of neurons**: Convolutional Neural Networks (ConvNets) constrain the architecture in a more sensible way. The layers of a ConvNet have neurons arranged in 3 dimensions: **width, height, depth**. 

**[DIAGRAM]** A regular 3-layer Neural Network vs. a ConvNet. The regular NN has columns of neurons. The ConvNet arranges its neurons in 3D (width, height, depth). The red input layer holds the image (width, height, 3 RGB channels). Every layer transforms the 3D input volume to a 3D output volume of neuron activations.

> **Key Insight**: A ConvNet is made up of Layers. Every Layer transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not have parameters.

---

## ConvNet Layers

A simple ConvNet is a sequence of layers. The three main types of layers are:
1. **Convolutional Layer**
2. **Pooling Layer**
3. **Fully-Connected Layer**

Example Architecture for CIFAR-10 classification: `[INPUT - CONV - RELU - POOL - FC]`
- **INPUT** `[32x32x3]` will hold the raw pixel values of the image.
- **CONV** layer will compute the output of neurons that are connected to local regions in the input. Might result in `[32x32x12]` if we use 12 filters.
- **RELU** layer will apply an elementwise activation function, such as $\max(0,x)$ thresholding at zero. Size remains `[32x32x12]`.
- **POOL** layer will perform downsampling along the spatial dimensions, resulting in `[16x16x12]`.
- **FC** layer computes the class scores, resulting in volume of size `[1x1x10]`.

---

## Convolutional Layer

The Conv layer is the core building block of a Convolutional Network.

### Local Connectivity

When dealing with high-dimensional inputs such as images, it is impractical to connect neurons to all neurons in the previous volume. Instead, we connect each neuron to only a local region of the input volume.

**[KEY CONCEPT] Receptive Field**: The spatial extent of this connectivity is a hyperparameter called the receptive field of the neuron (equivalently, the filter size). The extent of the connectivity along the depth axis is always equal to the depth of the input volume. 

**[DIAGRAM]** A 32x32x3 CIFAR-10 image. A volume of neurons in the first Conv layer. Each neuron is connected only to a local region spatially, but to the full depth (all 3 color channels). Five neurons along the depth look at the same region in the input (a depth column).

### Spatial Arrangement

Three hyperparameters control the size of the output volume: **depth, stride, and zero-padding**.

1. **Depth**: The number of filters we use. Each filter learns to look for something different in the input (e.g. oriented edges). A set of neurons looking at the same region of the input is called a **depth column** or **fibre**.
2. **Stride**: The jump size when we slide the filter. When stride is 1, we move the filters one pixel at a time. Stride 2 produces smaller output volumes spatially.
3. **Zero-padding**: Padding the input volume with zeros around the border. Allows us to control the spatial size of the output volumes (often to exactly preserve the spatial size).

**[FORMULA] Output Spatial Size**
We can compute the spatial size of the output volume $O$ as a function of:
- Input volume size: $W$
- Receptive field (filter) size: $F$
- Stride: $S$
- Zero padding: $P$

$$ O = \frac{W - F + 2P}{S} + 1 $$

If $O$ is not an integer, the stride is set incorrectly and the neurons cannot be tiled symmetrically across the input volume.

### Parameter Sharing

Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. 

**[KEY CONCEPT] Parameter Sharing**: We make an assumption: if one patch feature is useful to compute at some spatial position $(x, y)$, then it should also be useful to compute at a different position $(x_2, y_2)$. 

We denote a single 2-dimensional slice of depth as a **depth slice**. We constrain the neurons in each depth slice to use the same weights and bias. If an output volume is `[55x55x96]`, there are 96 depth slices, each of size `55x55`. If all neurons in a depth slice share weights, we only have 96 unique sets of weights. 

**[DIAGRAM]** Two distinct neurons in the output volume in the same depth slice looking at different spatial regions in the input. Because of parameter sharing, they share the exact same weights and bias. 

> **Key Insight**: Because of parameter sharing, the forward pass in each depth slice is mathematically a **convolution** of the neuron's weights with the input volume. This is why the sets of weights are called **filters** or **kernels**.

### Matrix Multiplication Implementation

The convolution operation can be efficiently implemented as a matrix multiplication using `im2col`.
1. The local regions in the input image are stretched out into columns. This operation is called **im2col**. 
2. The weights of the CONV layer are similarly stretched out into rows.
3. The convolution is evaluated as a large matrix multiplication `np.dot(W_row, X_col)`.
4. The result is reshaped back to its proper 3D output dimension.

**[CODE] im2col visualization (conceptual)**
```python
# W is [D, C, F, F] - D filters, C channels, FxF spatial
# X is [C, H, W] - C channels, HxW spatial
X_col = im2col(X, filter_size=F, stride=S) # shape: (C*F*F, O_H*O_W)
W_row = W.reshape(D, -1) # shape: (D, C*F*F)
out_col = np.dot(W_row, X_col) # shape: (D, O_H*O_W)
out = out_col.reshape(D, O_H, O_W) # Reshape to final volume
```

### Backpropagation

The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-flipped filters).

---

## Summary & Key Takeaways (Part 1)
- Regular NNs do not scale to images. ConvNets arrange neurons in 3D volumes.
- Convolutional layers rely on Local Connectivity and Parameter Sharing to reduce parameters.
- Output dimensions are determined by the formula $(W - F + 2P)/S + 1$.
- Convolutions can be computed highly efficiently using `im2col` and Matrix Multiplication.
- Part 2 will cover Pooling, Normalization, Fully-Connected layers and Architectures.
