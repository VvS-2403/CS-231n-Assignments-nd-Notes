# CS231n Lecture 10: Recurrent Neural Networks (Part 1)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Introduction: Why Sequential Processing](#introduction-why-sequential-processing)
- [Vanilla Recurrent Neural Networks (RNNs)](#vanilla-recurrent-neural-networks-rnns)
- [RNN Computational Graphs and Unrolling](#rnn-computational-graphs-and-unrolling)
- [Example: Character-Level Language Model](#example-character-level-language-model)
- [Backpropagation Through Time (BPTT)](#backpropagation-through-time-bptt)
- [Visualizing and Interpreting RNN Cells](#visualizing-and-interpreting-rnn-cells)
- [Application: Image Captioning](#application-image-captioning)

---

## Introduction: Why Sequential Processing

- In the previous lectures, we have extensively explored feedforward neural networks, particularly Convolutional Neural Networks (CNNs).

- These architectures are incredibly powerful for processing static data, such as images, where the input size is fixed (e.g., $224 \times 224 \times 3$) and the output size is also fixed (e.g., a vector of 1000 class probabilities).

- However, standard feedforward networks fall short when confronted with sequential data.

- Sequential data is characterized by an inherent ordering of elements and, crucially, a variable length. Examples include natural language (a sequence of words or characters), audio signals (a sequence of amplitudes over time), and video streams (a sequence of image frames).

- To process sequences, we need an architecture that maintains an internal "memory" or state of what it has seen so far, allowing it to process inputs of arbitrary lengths and incorporate historical context into future predictions. This brings us to Recurrent Neural Networks (RNNs).

- RNNs offer immense flexibility in their input and output configurations, allowing us to model a wide array of problems beyond the standard static mapping.

- We can visualize these configurations as computational graphs connecting inputs (typically denoted as $x$), hidden states ($h$), and outputs ($y$).

- **One-to-One**:
  - This is the standard feedforward neural network configuration.
  - It takes a single, fixed-size input and produces a single, fixed-size output.
  - For instance, basic image classification works this way. There is no sequential aspect here.

- **One-to-Many**:
  - In this setup, the network receives a single, fixed-size input but generates a sequence of outputs.
  - A prime example is image captioning. The input is a single image, and the output is a variable-length sequence of words describing the image.

- **Many-to-One**:
  - Here, the input is a sequence of variable length, but the output is a single, fixed-size vector.
  - Sentiment analysis of a sentence is a classic example. The input is a sequence of words (e.g., a movie review), and the network processes the entire sequence to output a single prediction, such as a positive or negative sentiment score.

- **Many-to-Many (Delayed or Seq2Seq)**:
  - In this configuration, both the input and the output are sequences, and their lengths do not necessarily have to match.
  - Machine translation is a typical application.
  - The network reads an input sequence in one language (e.g., English), summarizes it into an internal representation, and then generates an output sequence in another language (e.g., French).
  - The generation usually starts after the entire input sequence has been processed.

- **Many-to-Many (Synced)**:
  - This variation involves synchronized sequences of equal length.
  - For every input at time step $t$, the network produces an output at the same time step $t$.
  - Video classification on a frame-by-frame basis exemplifies this. Each frame in the video sequence is processed to predict an action class for that specific moment.

> **Key Insight**: RNNs allow us to break free from the constraints of fixed-size inputs and outputs, enabling the processing of sequences through a unified architecture that maintains a persistent hidden state over time.

- Even for data that is not inherently sequential, such as a static image, sequential processing can still be highly beneficial.

- For example, instead of processing an entire image in a single pass with a massive CNN, an attention-based RNN could take a series of "glimpses" at different parts of the image to gradually build an understanding and make a classification decision.

- Similarly, when generating a complex image, an RNN might draw it piece by piece rather than outputting all pixels simultaneously.

---

## Vanilla Recurrent Neural Networks (RNNs)

- The core idea of an RNN is to process a sequence of vectors $x_1, x_2, \dots, x_T$ by applying a recurrence formula at every time step.

- The network maintains an internal hidden state, denoted as $h_t$, which is updated based on both the new input at the current time step, $x_t$, and the previous hidden state, $h_{t-1}$.

- The general recurrence formula is defined as:

  $$h_t = f_W(h_{t-1}, x_t)$$

  - $h_t$: The new hidden state vector at time step $t$.
  - $f_W$: A function parameterized by a set of learnable weights $W$.
  - $h_{t-1}$: The old hidden state vector from the previous time step $t-1$.
  - $x_t$: The input vector at the current time step $t$.

- The most crucial characteristic of this formula is that the **same function $f_W$ and the same set of parameters $W$ are used at every single time step**.

- This parameter sharing is what allows the RNN to process sequences of arbitrary length and generalize across different positions in the sequence. It is directly analogous to how CNNs apply the same convolutional filters across all spatial locations of an image.

- The simplest realization of this concept is known as the "Vanilla RNN" or "Elman RNN" (named after Jeffrey Elman).

- In a Vanilla RNN, the hidden state $h_t$ is computed using a linear combination of the previous hidden state and the current input, followed by a non-linear activation function, typically the hyperbolic tangent ($\tanh$).

- The specific formula for the hidden state update in a Vanilla RNN is:

  $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$$

- Let's break down the components of this equation:
  - $W_{hh}$: The hidden-to-hidden weight matrix. It determines how the previous state influences the current state.
  - $W_{xh}$: The input-to-hidden weight matrix. It dictates how the current input modifies the state.
  - Both matrix multiplications produce vectors of the same dimension as $h_t$, which are added together (sometimes with a bias vector $b_h$, omitted here for simplicity).
  - $\tanh$: The hyperbolic tangent function squashes the resulting values to be strictly within the range $[-1, 1]$. This helps keep the hidden state values bounded as they are passed from step to step, preventing them from exploding to infinity immediately, although it doesn't solve all gradient issues, as we will see later.

- If the network is required to make a prediction at time step $t$, it computes an output vector $y_t$ directly from the current hidden state:

  $$y_t = W_{hy} h_t$$

  - $W_{hy}$: The hidden-to-output weight matrix. It maps the internal representation $h_t$ to the desired output format (e.g., raw logits for classification over a vocabulary).

- To initialize the process at the very first time step ($t=1$), where there is no previous hidden state $h_0$, it is standard practice to set $h_0$ to a vector of all zeros.

---

## RNN Computational Graphs and Unrolling

- To understand how an RNN processes a sequence, computes losses, and trains its weights, it is incredibly helpful to visualize its operations as a computational graph unrolled over time.

- Imagine the recurrence relation as a single node in a graph with a loop. At time $t$, it takes $x_t$ and $h_{t-1}$ to produce $h_t$ and $y_t$. When we "unroll" this graph, we explicitly draw out the computation for each time step from $1$ to $T$.

- In the unrolled graph, we see a sequence of nodes along a timeline.

- At $t=1$, the node receives $x_1$ and $h_0$ (usually zeros). It multiplies them by $W_{xh}$ and $W_{hh}$ respectively, applies $\tanh$, and outputs $h_1$. It may also use $h_1$ and $W_{hy}$ to compute $y_1$.

- At $t=2$, the next node receives $x_2$ and the newly computed $h_1$. Using the **exact same weight matrices** $W_{xh}$, $W_{hh}$, and $W_{hy}$, it computes $h_2$ and $y_2$.

- This process continues sequentially until time step $T$.

- Let's examine how this unrolled graph adapts to different configurations:

- **Many-to-Many**:
  - The graph is fully unrolled for $T$ steps.
  - At each step $t$, an output $y_t$ is generated.
  - If we are training a model (like predicting the next word), we have a ground truth target for each step.
  - We compute a loss $L_t$ (e.g., cross-entropy loss) by comparing $y_t$ to the target.
  - The total loss for the sequence is the sum (or average) of all individual losses: $L = \sum_{t=1}^T L_t$.

- **Many-to-One**:
  - The sequence of inputs $x_1 \dots x_T$ is fed into the network step by step.
  - However, we do not compute outputs or losses at intermediate steps.
  - Only at the final step $T$, after the network has digested the entire sequence, do we take the final hidden state $h_T$, compute the output $y$, and compare it to the single target label to calculate the loss $L$.
  - The intermediate hidden states $h_1 \dots h_{t-1}$ merely serve as memory carriers.

- **One-to-Many**:
  - A single input $x$ is provided, typically only at the first time step $t=1$.
  - For all subsequent steps $t > 1$, the input might be zero, or the network might feed its own previous output $y_{t-1}$ (or a sampled version of it) back as the input $x_t$ for the next step.
  - The RNN unrolls, producing a sequence of outputs $y_1 \dots y_T$ and corresponding losses.

- **Sequence-to-Sequence (Seq2Seq)**:
  - This is formed by connecting a many-to-one RNN (the encoder) to a one-to-many RNN (the decoder).
  - The encoder processes the input sequence $x_1 \dots x_T$ and condenses it into a single final hidden state $h_T$, often called the context vector.
  - This context vector is then passed as the initial hidden state to the decoder.
  - The decoder then unrolls over time to generate the output sequence.

> **Key Insight**: The unrolled computational graph makes an RNN look just like a very deep feedforward network, where each "layer" corresponds to a time step, and the weights are shared across all layers.

---

## Example: Character-Level Language Model

- To make the mechanics of an RNN crystal clear, let's walk through a concrete, numerical example of a character-level language model.

- The goal of this model is to learn the structure of text by predicting the next character in a sequence given the previous characters.

- Suppose we have an extremely limited vocabulary consisting of only four characters: `['h', 'e', 'l', 'o']`. We want to train the model on the single training sequence: the word `"hello"`.

- First, we need to represent our characters as vectors that the network can understand. We use one-hot encoding. Since our vocabulary size is 4, each character becomes a 4-dimensional vector:
  - `h`: $[1, 0, 0, 0]$
  - `e`: $[0, 1, 0, 0]$
  - `l`: $[0, 0, 1, 0]$
  - `o`: $[0, 0, 0, 1]$

- Our training sequence `"hello"` consists of an input sequence and a target sequence.
  - **Inputs ($x_1, x_2, x_3, x_4$)**: `"h"`, `"e"`, `"l"`, `"l"`.
  - **Targets**: We want the network to predict the next character. So the targets are `"e"`, `"l"`, `"l"`, `"o"`.

- Let's assume our hidden state size is 3 (so $h_t$ is a 3D vector). The matrices have shapes: $W_{xh}$ is $3 \times 4$, $W_{hh}$ is $3 \times 3$, and $W_{hy}$ is $4 \times 3$.

- **Forward Pass (Training):**

- 1. **Time step $t=1$**:
  - Input $x_1$ is one-hot vector for `"h"`: $[1, 0, 0, 0]$.
  - Previous state $h_0$ is initialized to $[0, 0, 0]$.
  - Compute hidden state: $h_1 = \tanh(W_{hh} h_0 + W_{xh} x_1)$. Notice that multiplying $W_{xh}$ by a one-hot vector essentially just selects the first column of $W_{xh}$. The $\tanh$ squashes the results. Let's say $h_1 = [0.3, -0.1, 0.9]$.
  - Compute output logits: $y_1 = W_{hy} h_1$. This gives a 4D vector of unnormalized scores, representing the network's confidence for the next character being 'h', 'e', 'l', or 'o'. Let's say $y_1 = [1.2, 2.5, 0.1, -0.5]$.
  - The target for $t=1$ is `"e"` (index 1). We apply a Softmax function to $y_1$ to get probabilities and compute the cross-entropy loss against the true label `"e"`. We want the score for `"e"` (2.5) to be higher than the others.

- 2. **Time step $t=2$**:
  - Input $x_2$ is one-hot vector for `"e"`: $[0, 1, 0, 0]$.
  - Previous state is now $h_1 = [0.3, -0.1, 0.9]$.
  - Compute hidden state: $h_2 = \tanh(W_{hh} h_1 + W_{xh} x_2)$. The state is updated based on the new input "e" and the context of having seen "h" ($h_1$).
  - Compute output logits: $y_2 = W_{hy} h_2$. Let's say $y_2 = [0.1, 0.2, 3.1, -1.0]$.
  - The target is `"l"`. We compute the loss. The network correctly gives a high score to "l" (3.1).

- 3. **Time steps $t=3$ and $t=4$**: We repeat this process. At $t=3$, input is `"l"`, target is `"l"`. At $t=4$, input is `"l"`, target is `"o"`.

- The total loss is the sum of the individual losses at $t=1, 2, 3, 4$.

- We then use backpropagation to update $W_{hh}$, $W_{xh}$, and $W_{hy}$ to minimize this loss. A brilliant, minimalistic implementation of this exact setup in raw NumPy (about 112 lines of code) is available in Andrej Karpathy's famous `min-char-rnn.py` gist.

- **Sampling (Test Time):**

- Once the model is trained, how do we generate novel text? We use an autoregressive generation process.

- 1. We start by feeding a seed character, say `"h"`, as $x_1$.

- 2. The network computes $h_1$ and output logits $y_1$.

- 3. We apply Softmax to $y_1$ to get a probability distribution over the vocabulary. For example, `[P(h)=0.05, P(e)=0.8, P(l)=0.1, P(o)=0.05]`.

- 4. We **sample** from this distribution. Let's say we sample `"e"`. (We don't always take the argmax, as sampling adds diversity to the generated text).

- 5. Crucially, we take the sampled character `"e"`, convert it to a one-hot vector, and feed it back in as the input $x_2$ for the next time step!

- 6. We repeat this process, letting the network's own output dictate its next input, thereby hallucinating a sequence of text character by character.

---

## Backpropagation Through Time (BPTT)

- Training an RNN involves computing the gradients of the total loss with respect to the weight matrices $W_{hh}$, $W_{xh}$, and $W_{hy}$.

- Because the RNN is unrolled over time, we perform backpropagation on this unrolled computational graph. This specialized process is called Backpropagation Through Time (BPTT).

- During the forward pass, we compute all hidden states $h_1 \dots h_T$, output logits $y_1 \dots y_T$, and losses $L_1 \dots L_T$ for the entire sequence.

- During the backward pass, the gradient of the total loss $L$ flows backward through the graph, starting from the end of the sequence at time $T$ and moving step-by-step back to time $t=1$.

- At any given time step $t$, the hidden state $h_t$ receives gradient signals from two distinct sources:
  - 1. **From the output at time $t$**: The gradient of $L_t$ with respect to $y_t$ flows through $W_{hy}$ into $h_t$.
  - 2. **From the next time step $t+1$**: The gradient flowing back from $h_{t+1}$ passes through the $f_W$ cell (specifically, through the $W_{hh}$ multiplication and the $\tanh$ derivative) to reach $h_t$.

- Because the weight matrices are shared across all time steps, the total gradient for a matrix like $W_{hh}$ is the sum of the gradients computed with respect to $W_{hh}$ at each individual time step.

### The Problem of Long Sequences: Truncated BPTT

- Consider training an RNN on an entire book, or a Wikipedia article consisting of tens of thousands of characters. Unrolling the graph for 100,000 steps has two catastrophic implications:

- 1. **Memory**: We must store the intermediate hidden states and activations for all 100,000 steps to compute the backward pass. This quickly exhausts GPU memory.

- 2. **Compute Time**: We must complete a forward pass of 100,000 steps before we can even begin to compute a single gradient update. Learning would be glacially slow.

- To solve this, we use **Truncated Backpropagation Through Time**. Instead of processing the entire sequence at once, we chop the long sequence into manageable chunks, say, of length 100.

- 1. We take the first chunk of 100 characters.

- 2. We perform a forward pass and a backward pass (BPTT) strictly within this 100-step window.

- 3. We update the network weights.

- 4. Now, we take the next chunk of 100 characters.

- 5. **Crucially**, we initialize the hidden state $h_0$ for this second chunk using the final hidden state $h_{100}$ from the previous chunk. The forward pass maintains an unbroken chain of hidden state memory across chunks!

- 6. However, when we perform the backward pass for the second chunk, we only backpropagate through the 100 steps of that specific chunk. We **truncate** the backward flow of gradients; we do not pass gradients back into the previous chunk.

- This approximation allows the model to theoretically maintain memory of the infinite past during the forward pass, while keeping memory and compute costs bounded during the backward pass.

---

## Visualizing and Interpreting RNN Cells

- When we train a large RNN (e.g., with a hidden state size of 512) on a massive corpus of text like the source code of the Linux kernel or the complete works of Shakespeare, the network learns complex representations. But can we understand what it has learned?

- Researchers have found that by visualizing the activations of individual dimensions (neurons) of the hidden state vector $h_t$ as the network reads through text, we can find remarkably interpretable "cells."

- Imagine reading a snippet of C code character by character. We track the value of, say, the 42nd element of the hidden vector $h_t$. We color-code the characters in the text: red if the value is positive and highly active, blue if it's negative and inactive.

- In a well-trained model, we can discover specialized cells that have learned high-level syntactic and semantic features completely unsupervised, just from the task of predicting the next character:
  - **Quote detection cell**: A cell that turns bright red the moment it reads an opening quote `"` and stays red until it reads the closing quote `"`, at which point it turns blue. It tracks whether the model is inside a string literal.
  - **Line length tracking cell**: A cell whose activation value slowly and steadily increases as it reads characters on a line. The moment it encounters a newline character `\n`, its activation drops drastically back to zero. It acts like a counter for line length.
  - **If-statement cell**: A cell that activates strongly specifically when the network is traversing the condition of an `if` statement (between the parentheses).
  - **Quote/Comment cell**: Similar to the quote cell, but specifically tracks whether the text is currently inside a block comment `/* ... */`.
  - **Code depth (indentation) cell**: A cell whose activation level correlates strongly with the depth of nested code blocks. It increases with every `{` and decreases with every `}`.

- These findings show that while the RNN is trained on a simple low-level objective (next character prediction), it learns to build powerful, abstract, and interpretable internal models of the data's structure.

---

## Application: Image Captioning

- We can marry the spatial understanding of Convolutional Neural Networks (CNNs) with the sequential modeling power of Recurrent Neural Networks (RNNs) to perform complex tasks mapping images to sequences. Image Captioning is the classic example.

- The goal is to input an image and output a natural language sentence describing the contents of the image.

- **The Architecture:**

- 1. **The Encoder (CNN)**:
  - We take a state-of-the-art CNN architecture, such as VGGNet or ResNet, that has been pre-trained on a large dataset like ImageNet.
  - We remove the final softmax classification layer.
  - We feed our input image through this CNN to extract a rich, high-level feature representation of the image.
  - For instance, in VGG-16, the output of the final fully connected layer (FC7) is a 4096-dimensional vector. Let's call this image feature vector $v$.

- 2. **The Decoder (RNN)**:
  - We use an RNN (usually an LSTM) to generate the sequence of words.

- **Connecting them:**
  - How do we condition the RNN on the image? We modify the vanilla recurrence formula to inject the image features $v$.
  - Before (Vanilla): $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1})$
  - Now (Conditioned): $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + W_{ih} v)$

- Here, $W_{ih}$ is a new weight matrix that projects the 4096-dimensional image vector into the hidden state dimension.

- Notice that the image features $v$ are added at every single time step.

- Alternatively, a simpler approach often used in practice is to project the image vector $v$ to match the hidden state size and use it directly as the initial hidden state $h_0$, rather than adding it at every step.

- **The Generation Process:**

- 1. **Initialization**: We feed a special `<START>` token as the first input $x_1$. The model computes $h_1$ (incorporating the image vector $v$) and predicts a probability distribution over the vocabulary for the first word.

- 2. **Sampling**: We sample a word from this distribution, say `"straw"`.

- 3. **Recurrence**: We take the sampled word `"straw"`, embed it into a vector, and feed it as the input $x_2$ for the next time step. The network updates to $h_2$ and predicts the next word, say `"hat"`.

- 4. **Termination**: The network continues this autoregressive loop until it eventually samples a special `<END>` token, signaling that the sentence is complete.

- This system can generate impressive, novel descriptions, such as "A cat sitting on a suitcase on the floor".

- However, it can also fail spectacularly, demonstrating that it doesn't truly "understand" physics or scene geometry.

- A classic failure case is an image of a woman holding a large, furry coat, which the model confidently captions as "A woman holding a cat in her hand" because the texture matches its learned representation of cats.
