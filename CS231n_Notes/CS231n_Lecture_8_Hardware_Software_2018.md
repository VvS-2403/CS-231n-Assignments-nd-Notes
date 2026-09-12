# CS231n Lecture 8: Deep Learning Hardware and Software
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
1. [Deep Learning Hardware](#deep-learning-hardware)
    - [CPU vs GPU](#cpu-vs-gpu)
    - [TPU](#tpu)
    - [CPU / GPU Communication](#cpu--gpu-communication)
2. [Deep Learning Software](#deep-learning-software)
    - [The Point of Deep Learning Frameworks](#the-point-of-deep-learning-frameworks)
    - [A Zoo of Frameworks](#a-zoo-of-frameworks)
3. [PyTorch](#pytorch)
    - [Fundamental Concepts](#fundamental-concepts)
    - [Tensors](#tensors)
    - [Autograd](#autograd)
    - [PyTorch: `nn` Module](#pytorch-nn-module)
    - [PyTorch: `optim`](#pytorch-optim)
    - [Defining New Modules](#defining-new-modules)
    - [DataLoaders](#dataloaders)
    - [Pretrained Models & Visdom](#pretrained-models--visdom)
4. [TensorFlow](#tensorflow)
    - [Neural Network Implementation](#neural-network-implementation)
    - [Optimizer and Loss](#optimizer-and-loss)
    - [TensorFlow Layers](#tensorflow-layers)
    - [High-Level Wrappers: Keras and Others](#high-level-wrappers-keras-and-others)
    - [Tensorboard & Distributed Execution](#tensorboard--distributed-execution)
5. [Static vs Dynamic Computation Graphs](#static-vs-dynamic-computation-graphs)
    - [Optimization & Serialization](#optimization--serialization)
    - [Conditionals & Loops](#conditionals--loops)
    - [Dynamic Graph Applications](#dynamic-graph-applications)
    - [Eager Execution in TensorFlow](#eager-execution-in-tensorflow)
    - [ONNX and Interoperability](#onnx-and-interoperability)
6. [Summary & Key Takeaways](#summary--key-takeaways)

---

## Section Headers from Slides

### Deep Learning Hardware

[KEY CONCEPT] **CPU (Central Processing Unit)**: The general-purpose brains of a computer. It has fewer cores, but each core is much faster and highly capable. It is great for sequential tasks.

[KEY CONCEPT] **GPU (Graphics Processing Unit)**: Specialized hardware originally designed for rendering graphics. It has thousands of cores, but each core is much slower and simpler ("dumber") than a CPU core. It is excellent for highly parallel tasks like matrix multiplications.

**CPU vs GPU Comparison (2018 Hardware)**:

| Feature | CPU (Intel Core i7-7700k) | GPU (NVIDIA GTX 1080 Ti) |
| :--- | :--- | :--- |
| **Cores** | 4 (8 threads with hyperthreading) | 3584 |
| **Clock Speed** | 4.2 GHz | 1.6 GHz |
| **Memory** | System RAM (e.g. 16-64 GB) | 11 GB GDDR5X |
| **Price** | $339 | $699 |
| **Speed** | ~540 GFLOPs FP32 | ~11.4 TFLOPs FP32 |

> **Key Insight**: GPUs provide a massive explosion in GigaFLOPs per dollar compared to CPUs, which fueled the deep learning revolution starting around 2012 (AlexNet).

In practice, for neural network operations (like training VGG or ResNet), GPUs are about **60x to 70x faster** than CPUs when not highly optimized, and optimized libraries like **cuDNN** provide another ~3x speedup over "unoptimized" CUDA code. 

**Programming GPUs**:
- **CUDA** (NVIDIA only): Write C-like code that runs directly on the GPU. Higher-level optimized APIs include cuBLAS, cuFFT, cuDNN, etc.
- **OpenCL**: Similar to CUDA, but runs on anything (including AMD GPUs and CPUs). However, it is usually slower on NVIDIA hardware.
- **HIP**: Automatically converts CUDA code to run on AMD GPUs.

### TPU

[KEY CONCEPT] **TPU (Tensor Processing Unit)**: Specialized hardware designed specifically by Google for deep learning.

- **NVIDIA TITAN V**: Has 5120 CUDA cores and 640 Tensor cores. Achieves ~14 TFLOPs FP32 and ~112 TFLOPs FP16. (Price: $2999)
- **Google Cloud TPU**: Has 64 GB HBM memory. Achieves ~180 TFLOPs. (Price: $6.50 per hour)
- **Google Cloud TPU Pod**: A massive cluster of 64 Cloud TPUs that can achieve ~11.5 PFLOPs of compute.

### CPU / GPU Communication

[DIAGRAM] An image of a computer case interior shows the CPU mounted on the motherboard and two massive GPUs mounted below. Arrows indicate that the **Model** resides on the GPU, while the **Data** is initially stored on the hard drive (HDD/SSD).

> **Key Insight**: If you aren’t careful, training can bottleneck on reading data from the disk and transferring it over the PCIe bus to the GPU!

**Solutions to Communication Bottlenecks**:
- Read all data into RAM.
- Use an SSD instead of an HDD.
- Use multiple CPU threads to prefetch data and feed it to the GPU asynchronously.

---

## Deep Learning Software

### A Zoo of Frameworks

There are many deep learning frameworks:
- **Caffe** (UC Berkeley) -> **Caffe2** (Facebook)
- **Torch** (NYU / Facebook) -> **PyTorch** (Facebook)
- **Theano** (U Montreal) -> **TensorFlow** (Google)
- **Others**: PaddlePaddle (Baidu), CNTK (Microsoft), MXNet (Amazon), Chainer, Deeplearning4j.

This lecture primarily focuses on **PyTorch** and **TensorFlow**.

### The Point of Deep Learning Frameworks

Why not just use numpy?
1. Quick to develop and test new ideas.
2. **Automatically compute gradients** via computational graphs.
3. Run it all efficiently on the GPU (wrapping cuDNN, cuBLAS, etc. under the hood).

---

## PyTorch

### Fundamental Concepts

- **Tensor**: Like a numpy array, but can run on the GPU.
- **Autograd**: A package for building computational graphs out of Tensors, and automatically computing gradients.
- **Module**: A neural network layer; may store state or learnable weights.

**PyTorch Versions**: The class focuses on PyTorch version 0.4.

### Tensors

PyTorch Tensors are just like numpy arrays, but they run on the GPU. The API is almost identical to numpy.

[CODE] Two-Layer Net using just PyTorch Tensors (Manual Gradients):
```python
import torch

device = torch.device('cuda:0') # Trivial to run on GPU - just change device!

N, D_in, H, D_out = 64, 1000, 100, 10
# Create random tensors for data and weights
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predictions and loss
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    # Backward pass: manually compute gradients
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    # Gradient descent step on weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

### Autograd

Instead of computing gradients manually, PyTorch's `autograd` can build a computational graph dynamically during the forward pass and compute all gradients for us during the backward pass.

[CODE] Two-Layer Net using PyTorch Autograd:
```python
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# requires_grad=True tells PyTorch to track operations on these tensors
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass builds the graph
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    # Automatically compute gradients of loss w.r.t w1 and w2
    loss.backward()
    
    # torch.no_grad() means "don't build a computational graph for this part"
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        # Zero out the gradients after updating (PyTorch accumulates gradients by default)
        # Methods ending in underscore (_) modify the Tensor in-place
        w1.grad.zero_()
        w2.grad.zero_()
```

### New Autograd Functions

You can define your own autograd functions by writing `forward` and `backward` methods, similar to the modular layers in assignments. You use a `ctx` object to "cache" values for the backward pass.

[CODE] Defining a custom ReLU autograd function:
```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x) # Cache input
        return x.clamp(min=0)
        
    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors # Retrieve cached input
        grad_input = grad_y.clone()
        grad_input[x < 0] = 0
        return grad_input

# Define a helper function to make it easy to use
def my_relu(x):
    return MyReLU.apply(x)

# Usage in forward pass
y_pred = my_relu(x.mm(w1)).mm(w2)
```

> **Key Insight**: In practice, you almost never need to define new autograd functions unless you need a highly custom backprop step (e.g. for a custom layer). Normal Python functions work perfectly fine because autograd tracks all basic PyTorch operations inside them.

### PyTorch: `nn` Module

The `torch.nn` package provides higher-level wrappers for working with neural networks. You can define your model as a sequence of layers (objects that hold learnable weights).

[CODE] Two-Layer Net using `torch.nn`:
```python
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Define model as a sequence of layers
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

learning_rate = 1e-2
for t in range(500):
    # Forward pass
    y_pred = model(x)
    # torch.nn.functional has useful helpers like loss functions
    loss = torch.nn.functional.mse_loss(y_pred, y)
    
    # Backward pass computes gradients for all model weights
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
        model.zero_grad()
```

### PyTorch: `optim`

Instead of manually updating weights with gradient descent, use an `optimizer` to apply different update rules like Adam, RMSProp, etc.

[CODE] Using `torch.optim`:
```python
# Use an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    
    loss.backward()
    
    # Update params and zero gradients
    optimizer.step()
    optimizer.zero_grad()
```

### Defining New Modules

A PyTorch `Module` is a neural network layer; it inputs and outputs Tensors. Modules can contain weights or other sub-modules. It is extremely common to define network components as Module subclasses.

[CODE] Creating a custom Module subclass:
```python
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        # Initializer sets up child modules
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        # Define forward pass using child modules
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
        # No need to define backward - autograd handles it!

model = TwoLayerNet(D_in, H, D_out)
```

You can even stack multiple instances of custom modules inside a `Sequential` block.

### DataLoaders

A `DataLoader` wraps a `Dataset` and provides minibatching, shuffling, and multithreading.

[CODE] Iterating over a DataLoader:
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# ... data setup ...
loader = DataLoader(TensorDataset(x, y), batch_size=8)
model = TwoLayerNet(D_in, H, D_out)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for epoch in range(20):
    # Iterate over loader to form minibatches
    for x_batch, y_batch in loader:
        y_pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Pretrained Models & Visdom

**Pretrained Models**: PyTorch makes it very easy to load pretrained models for transfer learning using `torchvision`.
```python
import torchvision
alexnet = torchvision.models.alexnet(pretrained=True)
vgg16 = torchvision.models.vgg16(pretrained=True)
resnet101 = torchvision.models.resnet101(pretrained=True)
```

**Visdom**: A visualization tool from Facebook Research to add logging to your code and visualize loss and statistics in a browser.

---

## TensorFlow

### Neural Network Implementation

Unlike PyTorch, where you run code and build graphs simultaneously, TensorFlow strictly separates the **definition** of the computational graph from its **execution**.

1. **Define the Graph**: Create placeholders and operations.
2. **Run the Graph**: Enter a `Session`, feed in numpy data, and get output numpy arrays.

[CODE] Two-Layer Net in TensorFlow:
```python
import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100

# 1. Define Computational Graph

# Create placeholders for input, weights, and targets
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal((D, H))) # Use Variable so weights persist in graph
w2 = tf.Variable(tf.random_normal((H, D)))

# Forward pass
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

# Tell TF to compute gradients
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Add assign operations to update w1 and w2 AS PART OF THE GRAPH!
learning_rate = 1e-5
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)
updates = tf.group(new_w1, new_w2) # Dummy node that depends on updates

# 2. Run the Graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Run once to init Variables
    values = {x: np.random.randn(N, D),
              y: np.random.randn(N, D)}
              
    for t in range(50):
        # Feed data and tell TF to compute the dummy "updates" node
        loss_val, _ = sess.run([loss, updates], feed_dict=values)
```
> **Key Insight**: If you don't explicitly request TensorFlow to run the `updates` node (or an optimizer operation) in `sess.run()`, the assign operations will not execute, and your loss will not go down!

### Optimizer and Loss

TensorFlow has built-in optimizers and predefined common losses.

[CODE] TensorFlow Optimizer and Loss:
```python
# Instead of manual gradients and assigns:
loss = tf.losses.mean_squared_error(y_pred, y)

optimizer = tf.train.GradientDescentOptimizer(1e-5)
updates = optimizer.minimize(loss)

# Later, in the session loop:
# Remember to execute the output of the optimizer!
loss_val, _ = sess.run([loss, updates], feed_dict=values) 
```

### TensorFlow Layers

Similar to PyTorch `nn.Module`, TF has high-level layers that automatically handle weight setup and initialization.

[CODE] TensorFlow Layers:
```python
init = tf.variance_scaling_initializer(2.0) # He initialization
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)
```

### High-Level Wrappers: Keras and Others

Because bare TensorFlow code can be highly verbose, many high-level wrappers have been built on top of it.
- **Keras**: Used to be a third-party wrapper but is now merged tightly into TensorFlow as `tf.keras`.
- **Other Wrappers**: `tf.estimator`, `tf.contrib.slim`, `Sonnet` (by DeepMind), `TFLearn`.

[CODE] Keras handles the entire training loop without needing manual Sessions or feed_dicts:
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(H, input_shape=(D,), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(D))

model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.SGD(lr=1e0))

# Keras manages the session and execution automatically
history = model.fit(x, y, epochs=50, batch_size=N)
```

### Tensorboard & Distributed Execution

- **Tensorboard**: An incredible tool for visualizing TensorFlow graphs. Add logging to your code to record loss, stats, etc., run the local server, and get beautiful interactive plots.
- **Distributed Version**: TensorFlow allows you to split a single computational graph across multiple machines to massively scale out training.

---

## Static vs Dynamic Computation Graphs

[KEY CONCEPT] **Static Graphs (TensorFlow, Caffe2)**: You build the graph structure once, and then you run it many times. The graph definition and execution are completely separate.
[KEY CONCEPT] **Dynamic Graphs (PyTorch, Chainer)**: Each forward pass builds a completely new graph structure. The graph building and computing happen simultaneously.

### Optimization & Serialization

**Static Graphs**:
- **Optimization**: Since the framework sees the entire graph before it runs, it can automatically optimize it (e.g. replacing a `Conv` operation followed by `ReLU` with a fused `Conv+ReLU` operation).
- **Serialization**: Once the graph is built, it can be serialized and run independently on different platforms (like C++) without needing the Python code that built it.

**Dynamic Graphs**:
- **Execution intertwined with building**: Since building and execution are mixed, you must keep the Python code around to execute the network.

### Conditionals & Loops

How do you implement conditional logic (e.g. `if z > 0`) or loops inside the graph?

**Dynamic (PyTorch)**: Use normal Python control flow (`if`, `for`). Since the graph is rebuilt every forward pass, the graph dynamically matches the execution path.
```python
# PyTorch
if z > 0:
    y = x.mm(w1)
else:
    y = x.mm(w2)
```

**Static (TensorFlow)**: You cannot use normal Python control flow because the graph is built before data flows through. You must use special framework-specific control flow nodes (`tf.cond`, `tf.foldl`).
```python
# TensorFlow
def f1(): return tf.matmul(x, w1)
def f2(): return tf.matmul(x, w2)
y = tf.cond(tf.less(z, 0), f1, f2)
```

### Dynamic Graph Applications

Dynamic graphs are incredibly powerful when the graph structure changes heavily depending on the input data:
1. **Recurrent Networks**: Graph size depends on the sequence length.
2. **Recursive Networks**: Tree-structured networks for tasks like parsing syntactic trees in NLP.
3. **Modular Networks**: e.g., Visual Question Answering where the question dictates the layout of the network modules dynamically.

### Eager Execution in TensorFlow

The lines between Static and Dynamic are blurring. TensorFlow 1.7 introduced **Eager Execution**, which allows dynamic graph building in TF.
```python
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

# Use GradientTape to scope dynamic operations, similar to PyTorch
with tfe.GradientTape() as tape:
    a = x * z
    b = a + z
    c = tf.reduce_sum(b)
grad_x = tape.gradient(c, [x])
```

### ONNX and Interoperability

**ONNX (Open Neural Network Exchange)**: An open-source standard for neural network models supported by PyTorch, Caffe2, CNTK, MXNet, etc.
- **Goal**: Make it easy to train a network in a dynamic, research-friendly framework (PyTorch), and then run it in a static, deployment-friendly framework (Caffe2).
- You export a PyTorch model to ONNX by tracing a dummy forward pass and saving the graph to a file.

---

## Summary & Key Takeaways

- **Deep Learning Hardware**: GPUs (NVIDIA) heavily dominate deep learning due to massive parallel core architectures. TPUs (Google) offer extremely fast specialized compute. Memory transfer bottlenecks (CPU to GPU) must be managed carefully using DataLoaders and RAM.
- **Deep Learning Software Frameworks**: Allow rapid prototyping, automatic gradient computation, and optimized GPU execution without writing raw CUDA code.
- **PyTorch (Dynamic Graphs)**: Clean, imperative API similar to numpy. Excellent for research, debugging, and dynamic models.
- **TensorFlow (Static Graphs)**: Separates graph definition from execution. Powerful ecosystem, highly optimized deployment (TensorBoard, TPUs, mobile), though raw TF can be verbose. High-level wrappers like **Keras** mitigate this.
- **Framework Recommendation**: PyTorch is strongly recommended for research and ease of use. TensorFlow is the safe bet for industry production and TPU support. Lines are blurring as TF adopts dynamic features (Eager Execution) and PyTorch improves static deployment capabilities (ONNX, Caffe2). 
- Next Lecture: CNN Architecture Case Studies.
