# CS231n Lecture 9: CNN Architectures
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Administrative](#administrative)
- [Review: LeNet-5](#review-lenet-5)
- [Case Study: AlexNet](#case-study-alexnet)
- [Case Study: VGGNet](#case-study-vggnet)
- [Case Study: GoogLeNet](#case-study-googlenet)
- [Case Study: ResNet](#case-study-resnet)
- [Comparing complexity...](#comparing-complexity)
- [Other architectures to know...](#other-architectures-to-know)
- [Summary: CNN Architectures](#summary-cnn-architectures)

---

## Section Headers from Slides

### Administrative

- **A2** due Wed May 2.
- **Midterm:** In-class Tue May 8. Covers material through Lecture 10 (Thu May 3).
- **Sample midterm** released on piazza.
- **Midterm review session:** Fri May 4 discussion section.

### Review: LeNet-5

[KEY CONCEPT] **LeNet-5 (LeCun et al., 1998)**: An early convolutional neural network used for digit recognition.
- Architecture: `[CONV-POOL-CONV-POOL-FC-FC]`
- Conv filters were 5x5, applied at stride 1
- Subsampling (Pooling) layers were 2x2 applied at stride 2

[DIAGRAM] An architectural diagram showing an input image passing through successive layers: Input -> Convolutions -> Subsampling -> Convolutions -> Subsampling -> Fully Connected -> Output.

### Case Study: AlexNet

[KEY CONCEPT] **AlexNet (Krizhevsky et al. 2012)**: The network that popularized CNNs in computer vision by winning ILSVRC 2012.

- **Architecture Overview**:
  - `CONV1 -> MAX POOL1 -> NORM1 -> CONV2 -> MAX POOL2 -> NORM2 -> CONV3 -> CONV4 -> CONV5 -> MAX POOL3 -> FC6 -> FC7 -> FC8`
- **Input**: 227x227x3 images
- **First layer (CONV1)**: 96 11x11 filters applied at stride 4.
  - Output volume size: $(227-11)/4+1 = 55$. Output volume: `[55x55x96]`
  - Parameters: $(11 \times 11 \times 3) \times 96 = 35\text{K}$
- **Second layer (POOL1)**: 3x3 filters applied at stride 2.
  - Output volume size: $(55-3)/2+1 = 27$. Output volume: `[27x27x96]`
  - Parameters: 0! (Pooling layers have no parameters)

**Full (simplified) AlexNet architecture:**
- `[227x227x3]` INPUT
- `[55x55x96]` CONV1: 96 11x11 filters at stride 4, pad 0
- `[27x27x96]` MAX POOL1: 3x3 filters at stride 2
- `[27x27x96]` NORM1: Normalization layer
- `[27x27x256]` CONV2: 256 5x5 filters at stride 1, pad 2
- `[13x13x256]` MAX POOL2: 3x3 filters at stride 2
- `[13x13x256]` NORM2: Normalization layer
- `[13x13x384]` CONV3: 384 3x3 filters at stride 1, pad 1
- `[13x13x384]` CONV4: 384 3x3 filters at stride 1, pad 1
- `[13x13x256]` CONV5: 256 3x3 filters at stride 1, pad 1
- `[6x6x256]` MAX POOL3: 3x3 filters at stride 2
- `[4096]` FC6: 4096 neurons
- `[4096]` FC7: 4096 neurons
- `[1000]` FC8: 1000 neurons (class scores)

> **Key Insight**: Historical note: Trained on GTX 580 GPU with only 3 GB of memory. Network spread across 2 GPUs, half the neurons (feature maps) on each GPU.
> CONV1, CONV2, CONV4, CONV5: Connections only with feature maps on same GPU.
> CONV3, FC6, FC7, FC8: Connections with all feature maps in preceding layer, communication across GPUs.

**Details/Retrospectives**:
- First use of ReLU
- Used Norm layers (not common anymore)
- Heavy data augmentation
- Dropout 0.5
- Batch size 128
- SGD Momentum 0.9
- Learning rate 1e-2, reduced by 10 manually when val accuracy plateaus
- L2 weight decay 5e-4
- 7 CNN ensemble: 18.2% -> 15.4% error

**ZFNet (Zeiler and Fergus, 2013)**
- Improved hyperparameters over AlexNet
- CONV1: change from (11x11 stride 4) to (7x7 stride 2)
- CONV3,4,5: instead of 384, 384, 256 filters use 512, 1024, 512
- ImageNet top 5 error: 16.4% -> 11.7%

### Case Study: VGGNet

[KEY CONCEPT] **VGGNet (Simonyan and Zisserman, 2014)**: Emphasizes deeper networks using smaller filters.

- **Small filters, Deeper networks**: 16 - 19 layers (VGG16Net / VGG19Net).
- Uses only 3x3 CONV stride 1, pad 1 and 2x2 MAX POOL stride 2.
- Why use smaller filters?
  - A stack of three 3x3 conv (stride 1) layers has the same effective receptive field as one 7x7 conv layer.
  - But it is deeper, meaning more non-linearities.
  - And fewer parameters: $3 \times (3^2 C^2)$ vs. $7^2 C^2$ for C channels per layer.

**Memory and Parameter Breakdown (VGG16)**
- INPUT: `[224x224x3]`. Memory: 150K. Params: 0.
- ... (Multiple CONV3 layers and POOL2 layers) ...
- FC: `[1x1x4096]`. Params: $7 \times 7 \times 512 \times 4096 = 102,760,448$
- TOTAL memory: 24M * 4 bytes ~= 96MB / image (only forward! ~*2 for bwd)
- TOTAL params: 138M parameters

> **Key Insight**: Most memory is consumed in the early CONV layers. Most parameters are concentrated in the late FC layers.

**Details**:
- ILSVRC’14 2nd in classification, 1st in localization.
- Similar training procedure as AlexNet. No LRN.
- Use VGG16 or VGG19 (VGG19 only slightly better, more memory).
- FC7 features generalize well to other tasks.

### Case Study: GoogLeNet

[KEY CONCEPT] **GoogLeNet (Szegedy et al., 2014)**: Focuses on computational efficiency using "Inception" modules.

- **Deeper networks, with computational efficiency**:
  - 22 layers
  - Efficient "Inception" module
  - No FC layers
  - Only 5 million parameters! 12x less than AlexNet.
  - ILSVRC’14 classification winner (6.7% top 5 error).

**Inception Module**:
- "Inception module": design a good local network topology (network within a network) and then stack these modules on top of each other.
- **Naive Inception module**:
  - Apply parallel filter operations on the input from the previous layer: Multiple receptive field sizes for convolution (1x1, 3x3, 5x5), and pooling (3x3).
  - Concatenate all filter outputs together depth-wise.
  - **Problem**: Computational complexity. The pooling layer preserves feature depth, which means total depth after concatenation can only grow at every layer! Very expensive compute.
- **Solution**: "Bottleneck" layers that use 1x1 convolutions to reduce feature depth.
  - [FORMULA] 1x1 CONV: preserves spatial dimensions, reduces depth! Projects depth to lower dimension.
  - Using bottleneck layers drastically reduces the number of operations (e.g., from 854M ops down to 358M ops for a given layer).

**Full Architecture**:
- Stem Network: `Conv-Pool -> 2x Conv-Pool`
- Stacked Inception Modules
- Classifier output (removed expensive FC layers!)
- Auxiliary classification outputs to inject additional gradient at lower layers `(AvgPool-1x1Conv-FC-FC-Softmax)`.
- 22 total layers with weights.

### Case Study: ResNet

[KEY CONCEPT] **ResNet (He et al., 2015)**: Introduces residual connections to train extremely deep networks.

- **Very deep networks using residual connections**:
  - 152-layer model for ImageNet.
  - ILSVRC’15 classification winner (3.57% top 5 error).
  - Swept all classification and detection competitions in ILSVRC’15 and COCO’15.
- **The Problem**: What happens when we continue stacking deeper layers on a "plain" convolutional neural network?
  - 56-layer model performs worse on both training and test error than a 20-layer model.
  - -> The deeper model performs worse, but it’s not caused by overfitting! It is an **optimization** problem.
- **Solution**: Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping.
  - Let $H(x)$ be the desired mapping. Fit $F(x) = H(x) - x$ instead, so $H(x) = F(x) + x$.
  - Use residual blocks with skip connections (identity).

**Architecture**:
- Stack residual blocks. Every residual block has two 3x3 conv layers.
- Periodically, double # of filters and downsample spatially using stride 2 (/2 in each dimension).
- Additional conv layer at the beginning.
- No FC layers at the end (Global average pooling layer after last conv layer, then FC 1000 to output classes).
- For deeper networks (ResNet-50+), use "bottleneck" layer to improve efficiency (similar to GoogLeNet): 1x1 conv down, 3x3 conv, 1x1 conv up.

**Training ResNet in practice**:
- Batch Normalization after every CONV layer.
- Xavier / He initialization.
- SGD + Momentum (0.9).
- Learning rate: 0.1, divided by 10 when validation error plateaus.
- Mini-batch size 256.
- Weight decay of 1e-5.
- No dropout used.

### Comparing complexity...

[DIAGRAM] Bubble chart comparing Operations (G-Ops) vs Top-1 accuracy vs Parameter count (bubble size).
- **VGG**: Highest memory, most operations.
- **GoogLeNet**: Most efficient.
- **AlexNet**: Smaller compute, still memory heavy, lower accuracy.
- **ResNet**: Moderate efficiency depending on model, highest accuracy.
- **Inception-v4**: Resnet + Inception.

### Other architectures to know...

- **Network in Network (NiN)** (Lin et al. 2014): Mlpconv layer with "micronetwork" (MLP/1x1 conv) within each conv layer to compute abstract features. Precursor to bottleneck layers.
- **Identity Mappings in Deep Residual Networks** (He et al. 2016): Improved block design, creates a more direct path for propagating information.
- **Wide Residual Networks** (Zagoruyko et al. 2016): Argues residuals are the important factor, not depth. Uses wider blocks.
- **ResNeXt** (Xie et al. 2016): Increases width of residual block through multiple parallel pathways ("cardinality"). Similar in spirit to Inception.
- **Stochastic Depth** (Huang et al. 2016): Randomly drop a subset of layers during each training pass to reduce vanishing gradients and training time. Bypass with identity function.
- **SENet** (Hu et al. 2017): "Squeeze-and-Excitation". Adds a "feature recalibration" module that learns to adaptively reweight feature maps using global avg pooling + 2 FC layers. ILSVRC'17 winner.
- **FractalNet** (Larsson et al. 2017): Ultra-deep neural networks without residuals. Fractal architecture with both shallow and deep paths to output.
- **DenseNet** (Huang et al. 2017): Dense blocks where each layer is connected to every other layer in feedforward fashion. Encourages feature reuse.
- **SqueezeNet** (Iandola et al. 2017): AlexNet-level accuracy with 50x fewer parameters and <0.5Mb model size using "Fire modules" (squeeze and expand layers).

**Neural Architecture Search (NAS)**:
- "Controller" network (RNN) that learns to design a good network architecture.
- Sample architecture, train it, get reward (accuracy), update controller using reinforcement learning.
- NASNet (Zoph et al. 2017): Uses NAS to find best cell structure on smaller CIFAR-10 dataset, then transfers architecture to ImageNet.

---

## Summary & Key Takeaways

- VGG, GoogLeNet, ResNet are all in wide use, available in model zoos.
- ResNet is current best default, also consider SENet when available.
- Trend towards extremely deep networks.
- Significant research centers around design of layer / skip connections and improving gradient flow.
- Efforts to investigate necessity of depth vs. width and residual connections.
- Even more recent trend towards meta-learning (NAS).
- **Next time:** Recurrent neural networks.
