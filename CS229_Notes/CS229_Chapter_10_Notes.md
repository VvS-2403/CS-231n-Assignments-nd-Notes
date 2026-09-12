# Part IV: Unsupervised Learning
## Chapter 10: Clustering and the k-means algorithm

### 1. Introduction to Clustering
In the clustering problem, we are given a training set $\{x^{(1)}, \dots, x^{(n)}\}$ and want to group the data into a few cohesive "clusters." 
- The data points are $x^{(i)} \in \mathbb{R}^d$.
- No labels $y^{(i)}$ are given. Therefore, this is an **unsupervised learning** problem.

### 2. The k-means Clustering Algorithm
The k-means clustering algorithm is an iterative method to partition the dataset. 

1. Initialize cluster centroids $\mu_1, \mu_2, \dots, \mu_k \in \mathbb{R}^d$ randomly.
2. Repeat until convergence: {
    For every $i$, set:
    $$c^{(i)} := \arg \min_j ||x^{(i)} - \mu_j||^2$$
    
    For each $j$, set:
    $$\mu_j := \frac{\sum_{i=1}^n 1\{c^{(i)} = j\}x^{(i)}}{\sum_{i=1}^n 1\{c^{(i)} = j\}}$$
}

#### Algorithm Details
- **$k$**: A parameter representing the number of clusters we want to find.
- **Cluster Centroids ($\mu_j$)**: Represent our current guesses for the positions of the centers of the clusters.
- **Initialization (Step 1)**: A common way to initialize is to randomly choose $k$ training examples and set the cluster centroids to be equal to the values of these $k$ examples. Other initialization methods are also possible.
- **Inner Loop (Step 2)**: The algorithm repeatedly carries out two steps:
    1. **Assignment step**: Assigning each training example $x^{(i)}$ to the closest cluster centroid $\mu_j$ (updating $c^{(i)}$).
    2. **Update step**: Moving each cluster centroid $\mu_j$ to the mean of the points assigned to it (updating $\mu_j$).

### 3. Visual Illustration (Figure 10.1 Description)
*Figure 10.1 illustrates the running of the k-means algorithm.*
- Training examples are shown as dots, and cluster centroids are shown as crosses.
- **(a)** Original dataset.
- **(b)** Random initial cluster centroids (in this instance, not chosen to be equal to two training examples).
- **(c-f)** Illustration of running two iterations of k-means:
    - In each iteration, we assign each training example to the closest cluster centroid (conceptually "painting" the training examples the same color as the cluster centroid to which it is assigned).
    - Then, we move each cluster centroid to the mean of the points assigned to it.

### 4. Convergence and the Distortion Function
Is the k-means algorithm guaranteed to converge? Yes, in a certain sense. Let us define the **distortion function** to be:

$$J(c, \mu) = \sum_{i=1}^n ||x^{(i)} - \mu_{c^{(i)}}||^2$$

- $J$ measures the sum of squared distances between each training example $x^{(i)}$ and the cluster centroid $\mu_{c^{(i)}}$ to which it has been assigned.
- It can be shown that k-means is exactly **coordinate descent** on the distortion function $J$.
- Specifically, the inner loop of k-means repeatedly:
    - Minimizes $J$ with respect to $c$ while holding $\mu$ fixed.
    - Minimizes $J$ with respect to $\mu$ while holding $c$ fixed.
- Therefore, $J$ must monotonically decrease, and the value of $J$ must converge.
- Usually, this implies that $c$ and $\mu$ will converge too. (In theory, it is possible for k-means to oscillate between a few different clusterings—i.e., different values for $c$ and/or $\mu$—that have exactly the same value of $J$, but this almost never happens in practice.)

### 5. Local Optima
- The distortion function $J$ is a **non-convex function**, so coordinate descent on $J$ is **not guaranteed to converge to the global minimum**.
- This means k-means can be susceptible to **local optima**.
- Very often, k-means works fine and finds very good clusterings despite this.
- If you are worried about getting stuck in bad local minima, a common practice is to **run k-means many times** using different random initial values for the cluster centroids $\mu_j$. Then, out of all the different clusterings found, pick the one that gives the **lowest distortion $J(c, \mu)$**.