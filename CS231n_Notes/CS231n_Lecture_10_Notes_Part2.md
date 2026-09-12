# CS231n Lecture 10: Recurrent Neural Networks (Part 2)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Image Captioning with Attention Mechanisms](#image-captioning-with-attention-mechanisms)
- [Application: Visual Question Answering (VQA)](#application-visual-question-answering-vqa)
- [Multilayer (Stacked) RNNs](#multilayer-stacked-rnns)
- [Long Short-Term Memory (LSTM) Networks](#long-short-term-memory-lstm-networks)
- [Gated Recurrent Units (GRU)](#gated-recurrent-units-gru)
- [Addressing Exploding Gradients: Gradient Clipping](#addressing-exploding-gradients-gradient-clipping)
- [Summary & Key Takeaways](#summary--key-takeaways)

---

## Image Captioning with Attention Mechanisms

- Standard image captioning compresses the entire image into a single, global feature vector $v$.

- This forces the RNN to try and extract every relevant detail from that bottleneck at every step of sentence generation. 

- **Attention mechanisms** revolutionized this approach by allowing the RNN to focus on different, specific parts of the image as it generates each word.

- Instead of taking the final 1D vector from the CNN, we take the output of a convolutional layer earlier in the network.

- This output is a 3D spatial grid of features, for example, a $14 \times 14 \times 512$ volume.

- We can think of this as $L$ distinct spatial locations ($L = 14 \times 14 = 196$), where each location is represented by a $D$-dimensional feature vector $v_i$ ($D = 512$).

- At each time step $t$, when the RNN is trying to generate a new word, it performs the following steps:

- 1. **Compute Attention Weights**:
  - The RNN's previous hidden state $h_{t-1}$ is used to query the image locations.
  - A small neural network (an MLP) takes $h_{t-1}$ and each spatial feature $v_i$, and outputs an unnormalized score.
  - These scores are passed through a Softmax to produce a probability distribution $p_1 \dots p_L$ over the $L$ spatial locations.
  - These $p_i$ values are the **attention weights**. If $p_5 = 0.9$, it means the network is paying 90% of its attention to spatial location 5.

- 2. **Compute Context Vector**:
  - The network calculates a weighted sum of the spatial feature vectors, using the attention weights.
  - This produces a single context vector $z$:
    $$z = \sum_{i=1}^L p_i v_i$$
  - This $z$ vector represents a "glimpse" of the image, heavily biased towards the areas the network decided were important for the current word.

- 3. **RNN Update**:
  - The context vector $z$, along with the input word $x_t$ and previous state $h_{t-1}$, is fed into the RNN to compute the new state $h_t$ and predict the next word.

- This approach is highly interpretable.

- When the model generates the word "bird" in the caption "A bird flying over a body of water", we can visualize the attention weights $p_i$ as a heat map over the image and literally see the network focusing intensely on the pixels containing the bird.

- There are two main variations:
  - **Soft attention**: As described above, computes a continuous weighted average of all locations. It is fully differentiable and can be trained with standard backpropagation.
  - **Hard attention**: The network samples exactly one discrete spatial location to look at, based on the probabilities $p_i$. Because this sampling process is non-differentiable, it must be trained using Reinforcement Learning techniques (like the REINFORCE algorithm). Soft attention is vastly more common in practice due to its ease of training.

---

## Application: Visual Question Answering (VQA)

- Visual Question Answering (VQA) pushes multimodal learning further.

- The input is an image *and* a natural language question about the image. The output is the answer.

- For example, given an image of a truck with a logo, the question might be "Q: What endangered animal is featured on the truck?", and the desired answer is "A: A bald eagle."

- This requires a sophisticated architecture that fuses information from both modalities.

- 1. **Image Encoding**:
  - The image is passed through a CNN to extract spatial feature maps, similar to the attention-based captioning model.

- 2. **Question Encoding**:
  - The question is treated as a sequence of words and passed through an RNN (usually an LSTM).
  - The final hidden state of this LSTM serves as a dense vector representation encoding the meaning of the question.

- 3. **Multimodal Attention and Fusion**:
  - The question vector is used to generate attention weights over the image's spatial features.
  - The network figures out "based on what the question is asking, where should I look in the image?"
  - The resulting context vector from the image is then fused (e.g., via concatenation or element-wise multiplication) with the question vector.

- 4. **Answer Prediction**:
  - This fused, multimodal representation is passed through a series of fully connected layers to a Softmax classifier, which outputs a probability distribution over a massive vocabulary of possible answers.

---

## Multilayer (Stacked) RNNs

- Just as we stack convolutional layers in a CNN to learn hierarchical representations (edges -> textures -> object parts), we can stack RNNs to increase their representational power. This creates a **Multilayer RNN**.

- In a multilayer RNN, depth is added in the vertical dimension. 

- The sequence of inputs $x_1 \dots x_T$ is fed into the first RNN layer.

- As the first layer unrolls over time, it produces a sequence of hidden states $h_1^{(1)} \dots h_T^{(1)}$.

- This entire sequence of hidden states acts as the input sequence to the second RNN layer. The second layer computes its own hidden states $h_1^{(2)} \dots h_T^{(2)}$ based on the outputs of the first layer.

- This stacking can continue for several layers. Only the hidden states of the top-most layer are used to compute the final outputs $y_t$.

- The intuition is that the lower RNN layers might model low-level, rapid temporal dynamics, while higher layers model slower, higher-level abstract concepts across the sequence. 

- Unlike CNNs, which can be hundreds of layers deep (e.g., ResNet-152), recurrent networks are rarely stacked very deep.

- A typical deep RNN might have 2 to 4 layers. Beyond that, optimization becomes exceedingly difficult, and the benefits plateau.

- If deeper networks are needed, techniques like skip connections between RNN layers are often required.

---

## Long Short-Term Memory (LSTM) Networks

- We mentioned earlier that Vanilla RNNs are simple but don't work very well in practice. The core reason lies in the dynamics of gradients during Backpropagation Through Time (BPTT). 

- Consider how the gradient flows backward from a loss at time $t$ to the hidden state at an early time $k$. The gradient must pass through the recurrence formula repeatedly.

- Specifically, the gradient of $h_t$ with respect to $h_{k}$ involves multiplying by the weight matrix $W_{hh}$ repeatedly $(t - k)$ times.

- If the largest singular value of $W_{hh}$ is strictly greater than 1, multiplying by it repeatedly causes the gradients to grow exponentially.
  - This is the **exploding gradient** problem, leading to numerical overflow (NaNs) and catastrophic failure during training.

- If the largest singular value of $W_{hh}$ is strictly less than 1, multiplying by it repeatedly causes the gradients to shrink exponentially toward zero.
  - This is the **vanishing gradient** problem.
  - The network completely fails to learn long-term dependencies because error signals from the future decay to zero before they can adjust weights corresponding to past events.

- The **Long Short-Term Memory (LSTM)** architecture, introduced by Hochreiter and Schmidhuber in 1997, was designed explicitly to combat the vanishing gradient problem.

- Instead of a single hidden state vector $h$, an LSTM maintains two separate states at each time step:
  - 1. $h_t$: The hidden state (which acts as the output of the cell).
  - 2. $c_t$: The **cell state** (an internal, private memory that doesn't get exposed directly to the outside world).

- The LSTM uses sophisticated gating mechanisms to carefully control what information is written to, erased from, and read from the cell state. 

- At time step $t$, the LSTM concatenates the previous hidden state $h_{t-1}$ and the current input $x_t$.

- It multiplies this large vector by a large weight matrix $W$ to compute four distinct gate vectors, each the same size as the hidden state:

  $$
  \begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = 
  \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{pmatrix}
  W \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix}
  $$

- Let's dissect these four components:
  - 1. $i$ (**Input gate**): Uses a sigmoid ($\sigma$) activation, squashing values to $[0, 1]$. It decides *whether* to write new information to the cell state. $0$ means "write nothing", $1$ means "write everything".
  - 2. $f$ (**Forget gate**): Uses a sigmoid ($\sigma$). It decides *whether* to erase or forget existing information from the previous cell state $c_{t-1}$. $0$ means "forget everything", $1$ means "remember everything".
  - 3. $o$ (**Output gate**): Uses a sigmoid ($\sigma$). It decides *how much* of the internal cell state $c_t$ should be revealed as the external hidden state $h_t$.
  - 4. $g$ (**Gate gate**, or block input): Uses a $\tanh$ activation, giving values in $[-1, 1]$. It proposes the actual *new candidate information* that could potentially be written to the cell state.

- *(Note on implementation: Instead of computing four separate matrix multiplications $W_i, W_f, W_o, W_g$, a common GPU optimization trick is to compute one massive matrix multiplication $W$ of size $4H \times (H+D)$, and then slice the resulting vector into the four chunks $i, f, o, g$.)*

- With these gates computed, the core magic of the LSTM happens in the cell state update:

  $$c_t = f \odot c_{t-1} + i \odot g$$

- Where $\odot$ represents element-wise multiplication. 

- This equation is beautiful. The new cell state $c_t$ is simply the old cell state $c_{t-1}$ multiplied by the forget gate $f$, plus the new candidate information $g$ scaled by the input gate $i$. 

- Finally, the external hidden state is updated:

  $$h_t = o \odot \tanh(c_t)$$

- The LSTM takes the internal cell state, squashes it with $\tanh$, and filters it through the output gate $o$ to produce $h_t$.

> **Key Insight**: Why does the LSTM solve vanishing gradients? Look at the gradient flow backward from $c_t$ to $c_{t-1}$. In a Vanilla RNN, this pathway involves multiplying by the matrix $W_{hh}$. In an LSTM, the pathway $c_t = f \odot c_{t-1} + \dots$ means the backpropagated gradient is simply multiplied element-wise by the forget gate vector $f$. There is no matrix multiplication by $W$ along this "highway"! Furthermore, if the network learns to set the forget gate $f \approx 1$, the gradient passes backward perfectly intact without decaying, acting as a Constant Error Carrousel. This additive interaction is fundamentally identical to the principle behind Residual Networks (ResNets).

---

## Gated Recurrent Units (GRU)

- The **Gated Recurrent Unit (GRU)** is a more recent (2014) and slightly simpler variant of the LSTM that performs very similarly in practice. 

- The GRU discards the separate cell state $c_t$ and relies only on a single hidden state vector $h_t$.

- It merges the input and forget gates into a single update gate, reducing the number of matrix multiplications and parameters, making it faster to compute.

- The equations for a GRU are:

  - 1. **Reset Gate ($r_t$)**: Decides how much of the past information to ignore.
       $$r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r)$$
  
  - 2. **Update Gate ($z_t$)**: Decides how much of the past information to keep vs. how much new information to mix in (acts as both input and forget gate).
       $$z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z)$$
  
  - 3. **Candidate Hidden State ($\tilde{h}_t$)**: Calculates the proposed new state, heavily influenced by the reset gate $r_t$. If $r_t$ is close to 0, it acts as if it's reading the first symbol of a sequence.
       $$\tilde{h}_t = \tanh(W_{xh} x_t + W_{hh}(r_t \odot h_{t-1}) + b_h)$$
  
  - 4. **Final Hidden State ($h_t$)**: A linear interpolation between the old state $h_{t-1}$ and the candidate state $\tilde{h}_t$, controlled by the update gate $z_t$.
       $$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

- The additive interaction $h_t = z_t \odot h_{t-1} + \dots$ preserves the uninterrupted gradient flow highway that solves the vanishing gradient problem, just like the LSTM.

- In practice, researchers often try both LSTM and GRU; neither is universally superior, though LSTMs are slightly more expressive and remain the default choice for many sequence modeling tasks.

---

## Addressing Exploding Gradients: Gradient Clipping

- While LSTMs and GRUs solve the *vanishing* gradient problem by creating additive paths, they do not guarantee protection against the *exploding* gradient problem.

- Even with LSTMs, repeated additions or specific weight configurations can cause the gradients flowing backward through the computational graph to blow up to astronomically large values.

- When a gradient explodes, taking a step in the direction of the gradient will throw the model weights completely out of the optimal region, often resulting in numerical overflow and the loss function instantly returning `NaN` (Not a Number). The training run is effectively ruined.

- To combat this, we use a remarkably simple but effective heuristic technique called **Gradient Clipping**.

- After we complete the backward pass and compute the gradients for all weights, but *before* we actually apply the parameter update (using SGD, Adam, etc.), we calculate the L2 norm of the gradient vector.

- If the norm of the gradient exceeds a predefined threshold (e.g., 5 or 10), we scale the entire gradient vector down so that its norm exactly equals the threshold.

```python
# Pseudo-code for gradient clipping
grad_norm = np.linalg.norm(gradients)
max_norm = 5.0
if grad_norm > max_norm:
    gradients = gradients * (max_norm / grad_norm)
```

- Gradient clipping acts as a safety net.

- It does not change the *direction* of the gradient vector, but it caps its *magnitude*.

- If the model encounters a steep cliff in the loss landscape that produces a massive gradient, clipping ensures that the optimizer only takes a sensible, fixed-size step in that direction, rather than catapulting the weights into oblivion. 

---

## Summary & Key Takeaways

- **RNNs process sequences** by maintaining a persistent hidden state that gets updated at every time step using shared parameters. This allows them to handle variable-length inputs and outputs flexibly (one-to-many, many-to-one, seq2seq).

- **Vanilla RNNs** use a simple linear transformation followed by $\tanh$, but are practically useless for long sequences due to the unstable gradient dynamics caused by repeated matrix multiplication during Backpropagation Through Time (BPTT).

- **Truncated BPTT** makes training on extremely long sequences feasible by restricting the backward pass to local chunks, saving memory and compute while allowing forward-pass memory to persist.

- **Interpretable cells** naturally emerge in the hidden states of RNNs trained on complex data like text or code, tracking features like quotes, brackets, and line lengths.

- **Image Captioning and VQA** demonstrate the power of fusing CNN spatial features with RNN sequential decoding, heavily augmented by **Attention mechanisms** that allow the network to selectively focus on relevant image regions during sequence generation.

- **LSTMs and GRUs** are the standard modern architectures for sequence modeling. They solve the vanishing gradient problem by introducing gating mechanisms (input, forget, output gates) that create additive gradient pathways (highways), bypassing the repeated matrix multiplications.

- **Gradient clipping** is a necessary and standard regularization technique used alongside LSTMs to prevent the surviving exploding gradients from causing catastrophic numerical instability during training.
