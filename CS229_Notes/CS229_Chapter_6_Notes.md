# CS229 - Support Vector Machines

## 6.1 Margins: intuition
* **Support Vector Machines (SVMs)** are among the best "off-the-shelf" supervised learning algorithms.
* This section talks about margins and the idea of separating data with a large "gap".
* Consider logistic regression, where the probability $p(y=1|x; \theta)$ is modeled by $h_\theta(x) = g(\theta^T x)$.
* We predict "1" on an input $x$ if and only if $h_\theta(x) \geq 0.5$, or equivalently, if and only if $\theta^T x \geq 0$.
* For a positive training example ($y=1$), the larger $\theta^T x$ is, the larger is $h_\theta(x) = p(y=1|x; \theta)$, and the higher our "confidence" that the label is 1. Informally, we are very confident that $y=1$ if $\theta^T x \gg 0$. Similarly, we are confident predicting $y=0$ if $\theta^T x \ll 0$.
* For a training set, we have found a good fit to the training data if we can find $\theta$ so that $\theta^T x^{(i)} \gg 0$ whenever $y^{(i)} = 1$, and $\theta^T x^{(i)} \ll 0$ whenever $y^{(i)} = 0$. This reflects a confident and correct set of classifications for all training examples. This goal leads to the notion of **functional margins**.
* **Diagram Description**: x's represent positive training examples, o's denote negative training examples. A decision boundary (separating hyperplane given by equation $\theta^T x = 0$) is shown. Three points A, B, and C are marked.
    * Point A is very far from the decision boundary. If asked to make a prediction for $y$ at A, we should be quite confident that $y=1$.
    * Point C is very close to the decision boundary. While on the $y=1$ side, a small change to the decision boundary could easily cause our prediction to be $y=0$. We are much more confident about our prediction at A than at C.
    * Point B lies in between these two cases.
* Generally, if a point is far from the separating hyperplane, we may be significantly more confident in our predictions. We want to find a decision boundary that allows us to make all correct and confident (far from the decision boundary) predictions. This is formalized using **geometric margins**.

## 6.2 Notation (option reading)
* Let's introduce new notation for classification.
* Target labels $y \in \{-1, 1\}$ (instead of $\{0, 1\}$).
* Parameters $w$, $b$ are used instead of $\theta$. The linear classifier is written as:
  $$h_{w,b}(x) = g(w^T x + b)$$
* Here, $g(z) = 1$ if $z \geq 0$, and $g(z) = -1$ otherwise.
* The explicit intercept term $b$ takes the role of previously used $\theta_0$, and $w$ takes the role of $[\theta_1 \dots \theta_d]^T$. We drop the convention of letting $x_0 = 1$.
* Note that our classifier will directly predict either $1$ or $-1$ (like the perceptron algorithm), without estimating $p(y=1)$ as an intermediate step (which logistic regression does).

## 6.3 Functional and geometric margins (option reading)
### Functional margins
* Given a training example $(x^{(i)}, y^{(i)})$, the **functional margin** of $(w, b)$ with respect to the training example is defined as:
  $$\hat{\gamma}^{(i)} = y^{(i)}(w^T x^{(i)} + b)$$
* If $y^{(i)}=1$, for the functional margin to be large (confident and correct prediction), we need $w^T x^{(i)} + b$ to be a large positive number.
* If $y^{(i)}=-1$, for the functional margin to be large, we need $w^T x^{(i)} + b$ to be a large negative number.
* A correct prediction implies $y^{(i)}(w^T x^{(i)} + b) > 0$. Hence, a large functional margin represents a confident and correct prediction.
* **Scale invariance issue**: Replacing $w$ with $2w$ and $b$ with $2b$ does not change $h_{w,b}(x)$ because $g(w^T x + b) = g(2w^T x + 2b)$, as $g$ only depends on the sign. However, this multiplies the functional margin by a factor of 2. We can make the functional margin arbitrarily large without changing anything meaningful by scaling $w$ and $b$. We might therefore impose a normalization condition such as $||w||_2 = 1$ (considering the margin of $(w/||w||_2, b/||w||_2)$).
* Given a training set $S = \{(x^{(i)}, y^{(i)}); i = 1, \dots, n\}$, the **functional margin of $(w, b)$ with respect to $S$** is the smallest functional margin among individual examples:
  $$\hat{\gamma} = \min_{i=1,\dots,n} \hat{\gamma}^{(i)}$$

### Geometric margins
* **Diagram Description**: The decision boundary for $(w, b)$ is shown as a line, with a vector $w$ orthogonal (at $90^\circ$) to it. A point A represents $x^{(i)}$ with $y^{(i)} = 1$. The distance from A to the decision boundary is denoted by $\gamma^{(i)}$, given by line segment AB.
* The vector $w/||w||$ is a unit-length vector pointing in the same direction as $w$. Point B on the decision boundary is given by:
  $$x^{(i)} - \gamma^{(i)} \cdot \frac{w}{||w||}$$
* Since point B lies on the decision boundary, it satisfies the equation $w^T x + b = 0$. Hence:
  $$w^T \left( x^{(i)} - \gamma^{(i)} \frac{w}{||w||} \right) + b = 0$$
* Solving for $\gamma^{(i)}$ yields:
  $$\gamma^{(i)} = \frac{w^T x^{(i)} + b}{||w||} = \left( \frac{w}{||w||} \right)^T x^{(i)} + \frac{b}{||w||}$$
* More generally, the **geometric margin** of $(w, b)$ with respect to a training example $(x^{(i)}, y^{(i)})$ is defined as:
  $$\gamma^{(i)} = y^{(i)} \left( \left( \frac{w}{||w||} \right)^T x^{(i)} + \frac{b}{||w||} \right)$$
* If $||w|| = 1$, then the functional margin equals the geometric margin.
* The geometric margin is **invariant to parameter rescaling**. Replacing $w$ with $2w$ and $b$ with $2b$ does not change the geometric margin. This lets us impose arbitrary scaling constraints on $w$, like $||w|| = 1$, without changing anything important.
* Given a training set $S$, the **geometric margin of $(w, b)$ with respect to $S$** is the smallest geometric margin on the training examples:
  $$\gamma = \min_{i=1,\dots,n} \gamma^{(i)}$$

## 6.4 The optimal margin classifier (option reading)
* Assume the training set is linearly separable. We want to find a decision boundary that maximizes the geometric margin, to reflect confident predictions.
* The optimization problem is:
  $$
  \begin{aligned}
  \max_{\gamma, w, b} \quad & \gamma \\
  \text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \geq \gamma, \quad i = 1, \dots, n \\
  & ||w|| = 1
  \end{aligned}
  $$
* Here we maximize $\gamma$ subject to each example having a functional margin of at least $\gamma$. The constraint $||w|| = 1$ ensures that functional margin equals geometric margin.
* The constraint $||w|| = 1$ is non-convex and difficult to solve. We transform the problem:
  $$
  \begin{aligned}
  \max_{\hat{\gamma}, w, b} \quad & \frac{\hat{\gamma}}{||w||} \\
  \text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \geq \hat{\gamma}, \quad i = 1, \dots, n
  \end{aligned}
  $$
* Here we maximize $\hat{\gamma}/||w||$ subject to functional margins being at least $\hat{\gamma}$. The geometric margin is $\gamma = \hat{\gamma}/||w||$. However, $\frac{\hat{\gamma}}{||w||}$ is still a non-convex objective function.
* We can introduce a **scaling constraint** on $w$ and $b$. We require the functional margin of $(w, b)$ with respect to the training set to be 1:
  $$\hat{\gamma} = 1$$
* Multiplying $w$ and $b$ by a constant multiplies the functional margin by the same constant, so this can be satisfied by rescaling $w, b$.
* Maximizing $\hat{\gamma}/||w|| = 1/||w||$ is the same as minimizing $||w||^2$. The optimization problem becomes:
  $$
  \begin{aligned}
  \min_{w, b} \quad & \frac{1}{2} ||w||^2 \\
  \text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \geq 1, \quad i = 1, \dots, n
  \end{aligned}
  $$
* This is a **convex quadratic objective** with only linear constraints, which can be efficiently solved using commercial **Quadratic Programming (QP)** code.
* To apply kernels and derive an efficient algorithm (SMO), we will formulate the dual of this optimization problem using Lagrange duality.

## 6.5 Lagrange duality (optional reading)
* Consider a constrained optimization problem:
  $$
  \begin{aligned}
  \min_{w} \quad & f(w) \\
  \text{s.t.} \quad & h_i(w) = 0, \quad i = 1, \dots, l
  \end{aligned}
  $$
* The **Lagrangian** is defined as:
  $$L(w, \beta) = f(w) + \sum_{i=1}^{l} \beta_i h_i(w)$$
* The $\beta_i$'s are the **Lagrange multipliers**. We find partial derivatives of $L$ with respect to $w_i$ and $\beta_i$ and set them to zero.
* Now consider the **primal optimization problem** with inequality constraints:
  $$
  \begin{aligned}
  \min_{w} \quad & f(w) \\
  \text{s.t.} \quad & g_i(w) \leq 0, \quad i = 1, \dots, k \\
  & h_i(w) = 0, \quad i = 1, \dots, l
  \end{aligned}
  $$
* The **generalized Lagrangian** is:
  $$L(w, \alpha, \beta) = f(w) + \sum_{i=1}^{k} \alpha_i g_i(w) + \sum_{i=1}^{l} \beta_i h_i(w)$$
* Here, $\alpha_i$ and $\beta_i$ are the Lagrange multipliers. Let's define the quantity:
  $$\theta_P(w) = \max_{\alpha, \beta : \alpha_i \geq 0} L(w, \alpha, \beta)$$
* If $w$ violates any of the primal constraints ($g_i(w) > 0$ or $h_i(w) \neq 0$), then:
  $$\theta_P(w) = \max_{\alpha, \beta : \alpha_i \geq 0} \left[ f(w) + \sum_{i=1}^{k} \alpha_i g_i(w) + \sum_{i=1}^{l} \beta_i h_i(w) \right] \quad (6.1)$$
  $$\theta_P(w) = \infty \quad (6.2)$$
* If $w$ satisfies all constraints, then $\theta_P(w) = f(w)$.
* Hence, $\theta_P(w)$ takes the value $f(w)$ if $w$ satisfies constraints, and $\infty$ otherwise. Thus:
  $$\min_w \theta_P(w) = \min_w \max_{\alpha, \beta : \alpha_i \geq 0} L(w, \alpha, \beta)$$
  has the same solution as the original primal problem. The optimal value of the primal problem is $p^* = \min_w \theta_P(w)$.
* The **dual optimization problem** is:
  $$\theta_D(\alpha, \beta) = \min_w L(w, \alpha, \beta)$$
  $$\max_{\alpha, \beta : \alpha_i \geq 0} \theta_D(\alpha, \beta) = \max_{\alpha, \beta : \alpha_i \geq 0} \min_w L(w, \alpha, \beta)$$
* The optimal value of the dual problem is $d^* = \max_{\alpha, \beta : \alpha_i \geq 0} \theta_D(\alpha, \beta)$.
* Weak duality: $d^* \leq p^*$.
* Under certain conditions, strong duality holds: **$d^* = p^*$**. We can then solve the dual problem instead of the primal. The conditions are:
  1. $f$ and $g_i$'s are convex.
  2. $h_i$'s are affine.
  3. The constraints $g_i$ are **strictly feasible** (there exists some $w$ such that $g_i(w) < 0$ for all $i$).
* If these conditions are met, there exist $w^*, \alpha^*, \beta^*$ which are solutions to the primal and dual problems, with $p^* = d^* = L(w^*, \alpha^*, \beta^*)$. They must satisfy the **Karush-Kuhn-Tucker (KKT) conditions**:
  $$
  \begin{aligned}
  \frac{\partial}{\partial w_i} L(w^*, \alpha^*, \beta^*) &= 0, \quad i = 1, \dots, d \quad &(6.3)\\
  \frac{\partial}{\partial \beta_i} L(w^*, \alpha^*, \beta^*) &= 0, \quad i = 1, \dots, l \quad &(6.4)\\
  \alpha_i^* g_i(w^*) &= 0, \quad i = 1, \dots, k \quad &(6.5)\\
  g_i(w^*) &\leq 0, \quad i = 1, \dots, k \quad &(6.6)\\
  \alpha_i^* &\geq 0, \quad i = 1, \dots, k \quad &(6.7)
  \end{aligned}
  $$
* Equation (6.5) is the **KKT dual complementarity condition**. It implies that if $\alpha_i^* > 0$, then $g_i(w^*) = 0$ (the constraint is active and holds with equality). This leads to the SVM having only a small number of **support vectors**.

## 6.6 Optimal margin classifiers: the dual form (option reading)
* The primal problem for the optimal margin classifier is:
  $$
  \begin{aligned}
  \min_{w, b} \quad & \frac{1}{2} ||w||^2 \quad &(6.8) \\
  \text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \geq 1, \quad i = 1, \dots, n
  \end{aligned}
  $$
* We can write the constraints as:
  $$g_i(w) = -y^{(i)}(w^T x^{(i)} + b) + 1 \leq 0$$
* By the KKT dual complementarity condition, $\alpha_i > 0$ only for training examples with functional margin exactly equal to 1 ($g_i(w) = 0$).
* **Diagram Description**: A maximum margin separating hyperplane is shown by a solid line. The points with the smallest margins are the ones closest to the decision boundary (one negative and two positive examples lying on dashed lines parallel to the decision boundary). Only the $\alpha_i$'s corresponding to these three training examples will be non-zero. These points are the **support vectors**. The fact that the number of support vectors is small is useful later.
* The Lagrangian is:
  $$L(w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum_{i=1}^{n} \alpha_i \left( y^{(i)}(w^T x^{(i)} + b) - 1 \right) \quad (6.9)$$
  (There are no $\beta_i$ multipliers because there are no equality constraints).
* To find the dual, we minimize $L$ with respect to $w$ and $b$:
  $$\nabla_w L(w, b, \alpha) = w - \sum_{i=1}^{n} \alpha_i y^{(i)} x^{(i)} = 0$$
  This implies:
  $$w = \sum_{i=1}^{n} \alpha_i y^{(i)} x^{(i)} \quad (6.10)$$
* For the derivative with respect to $b$:
  $$\frac{\partial}{\partial b} L(w, b, \alpha) = -\sum_{i=1}^{n} \alpha_i y^{(i)} = 0 \implies \sum_{i=1}^{n} \alpha_i y^{(i)} = 0 \quad (6.11)$$
* Plugging Equation (6.10) back into the Lagrangian (6.9) and simplifying using (6.11), we obtain:
  $$L(w, b, \alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} y^{(i)}y^{(j)}\alpha_i \alpha_j (x^{(i)})^T x^{(j)} - b \sum_{i=1}^{n} \alpha_i y^{(i)}$$
  Since the last term is zero (from 6.11):
  $$L(w, b, \alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} y^{(i)}y^{(j)}\alpha_i \alpha_j (x^{(i)})^T x^{(j)}$$
* The **dual optimization problem** is:
  $$
  \begin{aligned}
  \max_{\alpha} \quad & W(\alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} y^{(i)}y^{(j)}\alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \quad &(6.12) \\
  \text{s.t.} \quad & \alpha_i \geq 0, \quad i = 1, \dots, n \\
  & \sum_{i=1}^{n} \alpha_i y^{(i)} = 0
  \end{aligned}
  $$
* We solve this dual problem to find the optimal $\alpha$'s. We then use Equation (6.10) to find $w^*$. The optimal value for the intercept $b$ is found using the primal problem:
  $$b^* = -\frac{\max_{i:y^{(i)}=-1} w^{*T} x^{(i)} + \min_{i:y^{(i)}=1} w^{*T} x^{(i)}}{2} \quad (6.13)$$
* To make a prediction for a new input $x$, we calculate $w^T x + b$:
  $$
  \begin{aligned}
  w^T x + b &= \left( \sum_{i=1}^{n} \alpha_i y^{(i)} x^{(i)} \right)^T x + b \quad &(6.14)\\
  &= \sum_{i=1}^{n} \alpha_i y^{(i)} \langle x^{(i)}, x \rangle + b \quad &(6.15)
  \end{aligned}
  $$
* The prediction depends **only on the inner product** between $x$ and the support vectors (since $\alpha_i = 0$ for non-support vectors). This allows us to apply the **kernel trick**.

## 6.7 Regularization and the non-separable case (optional reading)
* If the data is not linearly separable, or to make the model less sensitive to outliers, we use $L_1$ regularization.
* **Diagram Description**: The left figure shows an optimal margin classifier. In the right figure, adding a single outlier in the upper-left region causes the decision boundary to make a dramatic swing, resulting in a much smaller margin.
* We reformulate the optimization problem to allow examples to have a functional margin less than 1, paying a cost $C\xi_i$:
  $$
  \begin{aligned}
  \min_{\gamma, w, b} \quad & \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i \\
  \text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i, \quad i = 1, \dots, n \\
  & \xi_i \geq 0, \quad i = 1, \dots, n
  \end{aligned}
  $$
* Parameter $C$ controls the relative weighting between maximizing the margin (small $||w||^2$) and ensuring most examples have functional margin at least 1.
* The Lagrangian is:
  $$L(w, b, \xi, \alpha, r) = \frac{1}{2} w^T w + C \sum_{i=1}^{n} \xi_i - \sum_{i=1}^{n} \alpha_i \left( y^{(i)}(x^T w + b) - 1 + \xi_i \right) - \sum_{i=1}^{n} r_i \xi_i$$
  where $\alpha_i, r_i \geq 0$ are Lagrange multipliers.
* The dual form of this problem is surprisingly similar to the separable case:
  $$
  \begin{aligned}
  \max_{\alpha} \quad & W(\alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} y^{(i)}y^{(j)}\alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \\
  \text{s.t.} \quad & 0 \leq \alpha_i \leq C, \quad i = 1, \dots, n \\
  & \sum_{i=1}^{n} \alpha_i y^{(i)} = 0
  \end{aligned}
  $$
* The only change in the dual problem is the constraint on $\alpha_i$: from $\alpha_i \geq 0$ to **$0 \leq \alpha_i \leq C$**.
* The calculation for $b^*$ has to be modified (Equation 6.13 is no longer valid).
* The KKT dual-complementarity conditions are:
  $$
  \begin{aligned}
  \alpha_i = 0 &\implies y^{(i)}(w^T x^{(i)} + b) \geq 1 \quad &(6.16) \\
  \alpha_i = C &\implies y^{(i)}(w^T x^{(i)} + b) \leq 1 \quad &(6.17) \\
  0 < \alpha_i < C &\implies y^{(i)}(w^T x^{(i)} + b) = 1 \quad &(6.18)
  \end{aligned}
  $$

## 6.8 The SMO algorithm (optional reading)
* The **SMO (sequential minimal optimization)** algorithm, developed by John Platt, solves the SVM dual problem efficiently.

### 6.8.1 Coordinate ascent
* Consider solving $\max_{\alpha} W(\alpha_1, \alpha_2, \dots, \alpha_n)$. The **coordinate ascent** algorithm works as follows:
  ```text
  Loop until convergence: {
      For i = 1, ..., n, {
          \alpha_i := \arg\max_{\hat{\alpha}_i} W(\alpha_1, ..., \alpha_{i-1}, \hat{\alpha}_i, \alpha_{i+1}, ..., \alpha_n).
      }
  }
  ```
* In the innermost loop, we hold all variables except $\alpha_i$ fixed, and reoptimize $W$ with respect to just $\alpha_i$.
* **Diagram Description**: The figure shows contours of a quadratic function to optimize. Coordinate ascent starts at (2, -2) and moves towards the global maximum by taking steps parallel to one of the axes, as only one variable is optimized at a time.

### 6.8.2 SMO
* We want to solve the dual problem:
  $$
  \begin{aligned}
  \max_{\alpha} \quad & W(\alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} y^{(i)}y^{(j)}\alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \quad &(6.19) \\
  \text{s.t.} \quad & 0 \leq \alpha_i \leq C, \quad i = 1, \dots, n \quad &(6.20) \\
  & \sum_{i=1}^{n} \alpha_i y^{(i)} = 0 \quad &(6.21)
  \end{aligned}
  $$
* If we hold $\alpha_2, \dots, \alpha_n$ fixed and try to take a coordinate ascent step for $\alpha_1$, we cannot make progress. Constraint (6.21) enforces:
  $$\alpha_1 y^{(1)} = -\sum_{i=2}^{n} \alpha_i y^{(i)}$$
  Since $y^{(1)} \in \{-1, 1\}$, multiplying by $y^{(1)}$ gives:
  $$\alpha_1 = -y^{(1)} \sum_{i=2}^{n} \alpha_i y^{(i)}$$
* Thus, $\alpha_1$ is completely determined by the other $\alpha_i$'s. We must update at least **two $\alpha$'s simultaneously** to satisfy the constraints.
* The SMO algorithm:
  ```text
  Repeat till convergence {
      1. Select some pair \alpha_i and \alpha_j to update next (using a heuristic that maximizes progress).
      2. Reoptimize W(\alpha) with respect to \alpha_i and \alpha_j, while holding all other \alpha_k's fixed.
  }
  ```
* Convergence is tested by checking if the KKT conditions (Equations 6.16-6.18) are satisfied within a tolerance (e.g., $0.01$ to $0.001$).
* To reoptimize $\alpha_1$ and $\alpha_2$ with $\alpha_3, \dots, \alpha_n$ fixed, we require:
  $$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = -\sum_{i=3}^{n} \alpha_i y^{(i)} = \zeta \quad (6.22)$$
* **Diagram Description**: The constraints on $\alpha_1$ and $\alpha_2$ restrict them to a box $[0, C] \times [0, C]$. They must also lie on the line $\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$. This line determines a lower-bound $L$ and an upper-bound $H$ on the permissible values for $\alpha_2$ such that $(\alpha_1, \alpha_2)$ lies in the box.
* Using equation (6.22), we can write $\alpha_1$ as a function of $\alpha_2$:
  $$\alpha_1 = (\zeta - \alpha_2 y^{(2)}) y^{(1)}$$
* Substituting $\alpha_1$ into the objective $W(\alpha)$, it becomes a quadratic function in just one variable $\alpha_2$: $a \alpha_2^2 + b \alpha_2 + c$.
* We can maximize this quadratic function easily by setting the derivative to zero. Let the resulting unclipped value be $\alpha_2^{\text{new,unclipped}}$.
* We then clip this value to ensure it lies in the valid interval $[L, H]$ due to the box constraint $0 \leq \alpha \leq C$:
  $$
  \alpha_2^{\text{new}} = \begin{cases}
  H & \text{if } \alpha_2^{\text{new,unclipped}} > H \\
  \alpha_2^{\text{new,unclipped}} & \text{if } L \leq \alpha_2^{\text{new,unclipped}} \leq H \\
  L & \text{if } \alpha_2^{\text{new,unclipped}} < L
  \end{cases}
  $$
* Finally, we use Equation (6.22) to find the corresponding optimal value for $\alpha_1^{\text{new}}$.