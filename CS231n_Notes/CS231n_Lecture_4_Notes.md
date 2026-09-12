# CS231n Lecture 4: Backpropagation

**Stanford CS231n | Spring 2018**

---

## Table of Contents
- [Introduction and Motivation](#introduction-and-motivation)
- [Derivatives and The Chain Rule](#derivatives-and-the-chain-rule)
- [Computational Graphs](#computational-graphs)
- [Intuition: Gates and Gradient Flow](#intuition-gates-and-gradient-flow)
- [Modularity & Staged Computation](#modularity--staged-computation)
- [Vectorized Backpropagation](#vectorized-backpropagation)

---

## Introduction and Motivation
**Backpropagation** is the method we use to efficiently compute the analytical gradients of highly complex expressions (like deep neural networks) by recursively applying the **chain rule** from calculus.

- **The Problem:** Given a complex function $f(x)$ (where $x$ is a vector of inputs and weights), we need to compute the gradient $\nabla f(x)$.
- **In Neural Networks:** The function $f$ is our loss function $L$. The inputs are our fixed training data $(x_i, y_i)$ and our variables are the weights $W$ and biases $b$. We want to compute $\nabla_W L$ so we can use gradient descent to update the parameters.

---

## Derivatives and The Chain Rule

### Simple Expressions
The derivative indicates the rate of change (sensitivity) of a function with respect to a specific variable for infinitesimally small changes.
- **Multiplication:** $f(x,y) = x \times y \implies \frac{\partial f}{\partial x} = y, \quad \frac{\partial f}{\partial y} = x$
- **Addition:** $f(x,y) = x + y \implies \frac{\partial f}{\partial x} = 1, \quad \frac{\partial f}{\partial y} = 1$
- **Max:** $f(x,y) = \max(x,y) \implies \frac{\partial f}{\partial x} = \mathbb{1}(x \ge y), \quad \frac{\partial f}{\partial y} = \mathbb{1}(y \ge x)$

### Compound Expressions & Chain Rule
Consider the compound function $f(x,y,z) = (x + y)z$.
We can break this down into intermediate variables:
1. $q = x + y$
2. $f = q \times z$

If we want the gradient of $f$ with respect to $x$, the **chain rule** dictates that we multiply the gradients along the path:
$$ \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \times \frac{\partial q}{\partial x} $$
Since $\frac{\partial f}{\partial q} = z$ and $\frac{\partial q}{\partial x} = 1$, the final gradient is $z \times 1 = z$.

---

## Computational Graphs
Any complex mathematical expression can be visualized as a **computational graph**, where nodes represent operations (gates) and edges represent the values passed between them.

### The Forward Pass
- Values flow from the inputs, through the operations, to the final output.
- Every gate computes its local output and caches any values it might need later.

### The Backward Pass (Backpropagation)
- Gradients flow backward from the final output (where the gradient is always 1, because $\frac{\partial f}{\partial f} = 1$) to the inputs.
- **The Core Equation:**  
  `Downstream Gradient = Upstream Gradient * Local Gradient`
  - **Upstream Gradient:** The gradient flowing into the node from the network ahead of it (computed by the chain rule previously).
  - **Local Gradient:** The derivative of the gate's output with respect to its immediate inputs (computed solely based on the gate's own mathematical operation).

> **Key Insight:** Backpropagation is a beautifully local process. A gate doesn't need to know anything about the macroscopic architecture of the neural network. It only needs to know how to compute its local gradient, and multiply it by the upstream gradient it receives.

---

## Intuition: Gates and Gradient Flow
If we view the circuit as "wanting" to increase the final output, the gates act as communication nodes sending signals backward about how inputs should change.

- **Add Gate ($+$): The Gradient Distributor.**
  - Local gradient is 1.
  - Multiplies the upstream gradient by 1 and passes it equally to all of its inputs.
  - *Intuition:* Addition affects the output equally, so both inputs receive the exact same gradient.
  
- **Max Gate ($\max$): The Gradient Router.**
  - Local gradient is 1 for the maximum input, and 0 for the others.
  - *Intuition:* The output value only depended on the largest input during the forward pass. Therefore, only the largest input can affect the output if tweaked.

- **Multiply Gate ($\times$): The Gradient Switcher/Scaler.**
  - Local gradient for $x$ is $y$, and for $y$ is $x$.
  - *Intuition:* The gradient passed to one input is scaled by the value of the *other* input. If a multiply gate receives inputs $x=1000$ and $y=1$, the gradient flowing to $y$ will be massive (1000x the upstream gradient), while the gradient flowing to $x$ will be tiny.

---

## Modularity & Staged Computation
Gates can be grouped together into macroscopic nodes, as long as you can calculate the analytical derivative of the entire group. 

**Example: The Sigmoid Function**
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
In a computational graph, this consists of multiple distinct gates: multiply by -1, exponential, add 1, invert.
However, if we know the analytical derivative of the whole expression:
$$ \frac{\partial \sigma}{\partial x} = \sigma(x)(1 - \sigma(x)) $$
We can group all those operations into a single "Sigmoid Gate". This dramatically simplifies implementation and reduces memory overhead because we don't need to cache the intermediate values of the individual operations.

**Staged Computation Pattern:**
When implementing backprop in code, it is extremely useful to break complex mathematical equations into staged intermediate variables during the forward pass, cache them, and then compute their gradients in reverse order during the backward pass.

---

## Vectorized Backpropagation
In real neural networks, inputs and weights are vectors, matrices, or high-dimensional tensors. The principles of backpropagation remain exactly the same, but the local gradients become **Jacobian matrices** (matrices of partial derivatives).

**Crucial Rules for Implementation:**
1. **Shape Matching:** The gradient of any variable must have the exact same shape as the variable itself. 
   - If $W$ is $[N \times D]$, then $\nabla_W L$ must be $[N \times D]$.
2. **Matrix Multiplication Gradients:** For an operation $Y = W \cdot X$:
   - The derivative with respect to $W$ involves a matrix multiplication with the transpose of $X$.
   - The derivative with respect to $X$ involves a matrix multiplication with the transpose of $W$.
3. **Dimensionality Checks:** The easiest way to derive vectorized backward passes is to analyze the dimensions. Since you know the shape of the upstream gradient and the shape of the local variable, there is usually only one mathematically valid way to arrange the transposes and matrix multiplications to produce a gradient of the correct shape.