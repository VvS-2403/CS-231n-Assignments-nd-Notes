# CS231n Lecture 2: Image Classification Pipeline

**Stanford CS231n | Spring 2018**

---

## Table of Contents
- [Image Classification Overview](#image-classification-overview)
- [The Data-Driven Approach](#the-data-driven-approach)
- [Nearest Neighbor Classifier](#nearest-neighbor-classifier)
- [k-Nearest Neighbor Classifier](#k-nearest-neighbor-classifier)
- [Hyperparameter Tuning & Validation](#hyperparameter-tuning--validation)
- [Applying k-NN in Practice](#applying-k-nn-in-practice)

---

## Image Classification Overview

### Motivation
Image classification is the task of assigning an input image one label from a fixed set of categories. It is a core Computer Vision problem that forms the foundation for more complex tasks like object detection and segmentation.

### The Semantic Gap and Challenges
To a computer, an image is a large 3D array of numbers (e.g., $248 \times 400 \times 3$ for RGB), with integers ranging from 0 to 255. Translating this grid of numbers into a semantic label (like "cat") is difficult due to several variations:
- **Viewpoint Variation:** A single object can be oriented in many different ways.
- **Scale Variation:** Visual classes exhibit extreme size differences.
- **Deformation:** Non-rigid objects can deform into unusual shapes.
- **Occlusion:** Only a small portion of the object may be visible.
- **Illumination Conditions:** Drastic pixel-level changes occur under different lighting.
- **Background Clutter:** Objects often blend into their surroundings.
- **Intra-class Variation:** A single class (e.g., "chair") encompasses many different structural appearances.

A robust model must remain invariant to these variations while being sensitive enough to distinguish between different classes.

---

## The Data-Driven Approach

Instead of writing explicit rules for recognizing every object (which is practically impossible), we rely on a **data-driven approach**.
1. **Input:** Provide the computer with a training dataset of $N$ images, each labeled with one of $K$ classes.
2. **Learning:** Develop an algorithm to learn the visual appearance of each class from these examples.
3. **Evaluation:** Test the trained classifier on a novel set of unseen images and compare predictions to the ground truth labels.

**Dataset Example: CIFAR-10**
- Contains 60,000 images ($32 \times 32 \times 3$).
- 10 distinct classes.
- Split into a 50,000-image training set and a 10,000-image test set.

---

## Nearest Neighbor Classifier

As a first approach, the Nearest Neighbor (NN) classifier simply memorizes the entire training dataset. At test time, it compares the test image to every training image and predicts the label of the closest match.

### Distance Metrics
To quantify the difference between two images ($I_1, I_2$), we use distance functions:

- **L1 Distance (Manhattan Distance):**  
  $d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$
  - Computes pixel-wise absolute differences and sums them.
  - Coordinate-dependent.

- **L2 Distance (Euclidean Distance):**  
  $d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$
  - Penalizes large pixel differences much more heavily than L1 distance.
  - Prefers many small disagreements over one massive disagreement.

### Pros and Cons of Nearest Neighbor
- **Pros:** 
  - Extremely simple to implement.
  - Training is $O(1)$ (just storing data).
- **Cons:**
  - Prediction is heavily bottlenecked at $O(N)$ since a test image must be compared to every training image. In practice, we want fast testing and are okay with slow training—the exact opposite of NN.
  - **Curse of Dimensionality:** Pixel-based distances are terrible at capturing perceptual similarity (e.g., shifting an image by one pixel drastically changes L2 distance but doesn't change the semantic content). High dimensions require exponentially massive datasets to densely populate the space.

---

## k-Nearest Neighbor Classifier (k-NN)

Using just the single closest neighbor ($k=1$) leads to disjointed decision boundaries and overfitting to noise (e.g., a green point deep inside a cluster of blue points creates an incorrect "island").

**The k-NN Solution:**
Instead of taking the single closest image, find the top $k$ closest images and have them **vote** on the final label.
- Higher values of $k$ smooth out the decision boundaries.
- It leads to better resistance to outliers and better **generalization**.

---

## Hyperparameter Tuning & Validation

**Hyperparameters** are design choices about the algorithm that are not learned from the data (e.g., the value of $k$ or the choice between L1/L2 distance).

### The Golden Rule of Test Data
> **Never use the test set to tune hyperparameters.**

Tuning on the test set causes the model to **overfit** to it, destroying its ability to act as an unbiased estimator of real-world performance. The test set must be touched *only once* at the very end of the pipeline.

### Validation Sets
Instead of using the test set, we split our original training data into two portions:
1. **Training Set:** Used to build the model (e.g., 49,000 images).
2. **Validation Set:** A "fake" test set used to tune hyperparameters (e.g., 1,000 images).

**Workflow:** Train models with different hyperparameters on the training set, evaluate them on the validation set, and pick the hyperparameters that yield the highest validation accuracy.

### Cross-Validation
When datasets are very small, a single validation split might be too noisy.
- Divide the training data into $F$ equal folds (e.g., 5 folds).
- Iterate through each fold: use 1 fold for validation and the other 4 for training.
- Average the validation performance across all folds.
- *Note:* This is computationally expensive and generally skipped in deep learning in favor of a single large validation split.

---

## Applying k-NN in Practice
If you must use k-NN, follow this robust pipeline:
1. **Preprocess data:** Normalize features to zero mean and unit variance (less critical for homogeneous image pixels).
2. **Dimensionality Reduction:** Use PCA or Random Projections to compress high-dimensional raw images before applying k-NN.
3. **Split Data:** Allocate 70-90% to training and the rest to validation (or use cross-validation).
4. **Tune Hyperparameters:** Evaluate across many values of $k$ and both L1/L2 distances.
5. **Optimize Speed:** Use Approximate Nearest Neighbor (ANN) libraries like FLANN to speed up $O(N)$ lookup times at a slight accuracy cost.
6. **Final Evaluation:** Evaluate your best hyperparameter configuration strictly once on the test set.