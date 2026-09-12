# CS231n Lecture 5: Convolutional Neural Networks (Part 2)
**Stanford CS231n | Spring 2017**  
**Instructors:** Fei-Fei Li, Justin Johnson, Serena Yeung

---

## Table of Contents
- [Pooling Layer](#pooling-layer)
- [Normalization Layer](#normalization-layer)
- [Fully-Connected Layer](#fully-connected-layer)
- [Converting FC Layers to CONV Layers](#converting-fc-layers-to-conv-layers)
- [ConvNet Architectures](#convnet-architectures)
  - [Layer Patterns](#layer-patterns)
  - [Layer Sizing Patterns](#layer-sizing-patterns)
  - [Case Studies](#case-studies)
  - [Computational Considerations](#computational-considerations)

---

## Pooling Layer

It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. 

**[KEY CONCEPT] Pooling**: The function of the Pooling layer is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. 

The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. 

**[DIAGRAM]** Max pooling downsampling a `224x224x64` volume to a `112x112x64` volume using a filter size of 2 and a stride of 2. The depth is preserved.
**[DIAGRAM]** A 4x4 matrix undergoing max pooling with a 2x2 filter and stride 2. Each 2x2 colored block outputs its maximum value to a 2x2 output matrix. 

**[FORMULA] Pooling Output Size**
For an input volume of $W_1 \times H_1 \times D_1$, and pooling hyperparameters spatial extent $F$ and stride $S$:
- $W_2 = (W_1 - F)/S + 1$
- $H_2 = (H_1 - F)/S + 1$
- $D_2 = D_1$

> **Key Insight**: The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2. This discards 75% of the activations. Average pooling was historically used but has fallen out of favor compared to Max pooling.

---

## Normalization Layer

**[KEY CONCEPT] Local Response Normalization (LRN)**: Many types of normalization layers have been proposed (like LRN) intended to mimic biological inhibition. 
However, these layers have fallen out of favor since they don't seem to contribute significantly to performance in practice, while adding complexity. (Note: Batch Normalization, covered in later lectures, is widely used instead).

---

## Fully-Connected Layer

Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

---

## Converting FC Layers to CONV Layers

It is worth noting that the only difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters. Both layers compute dot products, so their functional form is identical. Therefore, it's possible to convert between them.

**[KEY CONCEPT] FC to CONV Conversion**: 
For any FC layer, we can convert it to a CONV layer. 
For example, an FC layer looking at a `[7x7x512]` volume with 4096 outputs can be replaced by a CONV layer with $F=7$, $P=0$, $S=1$, and 4096 filters. 
The output will be `[1x1x4096]`.

**Why convert FC to CONV?** 
If we replace all FC layers with CONV layers in a network (like AlexNet), we can slide the network across larger images in a single forward pass, rather than cropping the image and running the network multiple times. This is much more computationally efficient.

---

## ConvNet Architectures

### Layer Patterns

The most common ConvNet architecture follows the pattern:
`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

Where `*` indicates repetition, and `POOL?` indicates an optional pooling layer. Usually, $N \ge 0$ (and usually $N \le 3$), $M \ge 0$, $K \ge 0$ (and usually $K < 3$).

> **Key Insight**: Prefer a stack of small filter CONV layers to one large receptive field CONV layer. A stack of three 3x3 CONV layers has the same effective receptive field as one 7x7 CONV layer. However, the 3x3 stack uses fewer parameters and applies more non-linearities (ReLUs), allowing it to express more complex features.

### Layer Sizing Patterns

- **Input layer**: Should be divisible by 2 many times (e.g., 32, 64, 96, 224, 384, 512).
- **CONV layers**: Should use small filters (e.g., 3x3 or at most 5x5), a stride of $S=1$, and padding to preserve spatial dimensions (e.g., $P=1$ for $F=3$).
- **POOL layers**: Generally use max pooling with $F=2$, $S=2$.

### Case Studies

**[KEY CONCEPT] Important architectures in history:**
1. **LeNet (1998)**: The first successful applications of ConvNets (by Yann LeCun) used for reading zip codes and digits.
2. **AlexNet (2012)**: The first work that popularized ConvNets in Computer Vision (by Alex Krizhevsky, Ilya Sutskever and Geoff Hinton). Featured a very similar architecture to LeNet but was deeper, bigger, and featured Conv Layers stacked on top of each other.
3. **ZF Net (2013)**: The ILSVRC 2013 winner, an improvement on AlexNet by tweaking the architecture hyperparameters (expanding the size of middle convolutional layers and making the stride and filter size on the first layer smaller).
4. **GoogLeNet (2014)**: The ILSVRC 2014 winner. Introduced the **Inception Module**, which dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Also removed FC layers, replacing them with Average Pooling to save parameters.
5. **VGGNet (2014)**: The runner-up in ILSVRC 2014. Showed that depth is critical. VGG16 contains 16 CONV/FC layers, exclusively using 3x3 convolutions with stride 1 and pad 1, and 2x2 max pooling with stride 2. Cons: extremely heavy in parameters (140M) and memory usage.
6. **ResNet (2015)**: Residual Network developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special **skip connections** and a heavy use of batch normalization, missing FC layers at the end. State of the art ConvNet models.

### Computational Considerations

The largest bottleneck to be aware of when constructing ConvNet architectures is the memory bottleneck. Three major sources of memory usage:
1. **Intermediate volume sizes**: The raw number of activations at every layer, and their gradients. 
2. **Parameter sizes**: The weights of the network, their gradients, and a step cache if using momentum/Adagrad/RMSProp. 
3. **Miscellaneous**: Memory for image data batches, etc.

To estimate memory: `Total bytes = (Number of values) * 4 bytes/value (for float32)`

If your network doesn't fit in memory, common practice is to decrease the batch size, since most memory is consumed by the activations.

---

## Summary & Key Takeaways (Part 2)
- Pooling layers (usually Max Pooling) aggressively downsample the spatial dimensions to reduce parameters and computation.
- FC layers can be converted to CONV layers for efficient evaluation of larger images.
- A stack of small filters (3x3) is preferred over a single large filter (7x7) due to fewer parameters and more non-linearities.
- Notable historical architectures: LeNet, AlexNet, ZFNet, GoogLeNet, VGGNet, ResNet.
- Memory constraints are a significant computational consideration, primarily driven by activation volumes early in the network.

