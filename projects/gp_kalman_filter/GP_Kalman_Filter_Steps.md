# Local Gaussian Process Regression Filter with Bayesian Fusion

This document outlines the mathematical steps for a non-parametric time-series filter. The filter uses a sliding-window Gaussian Process (GP) to predict the state and a Bayesian (Kalman) update to fuse measurements.

## 1. Notation and Setup

- $t$: Current time step.
- $N$: Length of the sliding window (number of past data points).
- $\mathbf{T} = [t-N, \dots, t-1]^T$: Vector of past time steps in the buffer.
- $\mathbf{z} = [z_{t-N}, \dots, z_{t-1}]^T$: Vector of past **raw measurements** stored in the buffer.
- $\mathbf{R_{past}} = [R_{t-N}^*, \dots, R_{t-1}^*]^T$: Vector of their corresponding **inflated measurement variances** (used in the GP noise matrix).
- $z_t$: Measurement at time $t$.
- $R_t$: Base measurement variance at time $t$ (from keypoint confidence).
- $\lambda$: Variance inflation factor (innovation gating scale).
- $k(x, x')$: Covariance kernel function (e.g., RBF or Matérn).

## 2. Step 1: The Prediction Step (Heteroscedastic GP Time Update)

To predict the signal at time $t$, we fit a Gaussian Process to the recent window of $N$ raw measurements. Because measurements have varying confidence, we replace the standard uniform noise assumption ($\sigma_n^2 \mathbf{I}$) with a diagonal matrix of the model predicted measurement variances. This ensures the GP trusts high-confidence measurements more than low-confidence ones.

Let:

- $\mathbf{K} = k(\mathbf{T}, \mathbf{T})$ be the $N \times N$ kernel matrix of the past time steps.
- $\mathbf{k}_* = k(\mathbf{T}, t)$ be the $N \times 1$ cross-covariance vector between past steps and the current step.
- $k_{**} = k(t, t)$ be the prior variance at time $t$.
- $\mathbf{V} = \text{diag}(R_{t-N}^*, \dots, R_{t-1}^*)$ be the heteroscedastic noise matrix.

The predicted mean ($\mu_{t|t-1}$) and predicted variance ($\Sigma_{t|t-1}$) at time $t$ are:

$$\mu_{t|t-1} = \mathbf{k}_*^T (\mathbf{K} + \mathbf{V})^{-1} \mathbf{z}$$

$$\Sigma_{t|t-1} = k_{**} - \mathbf{k_*}^T (\mathbf{K} + \mathbf{V})^{-1} \mathbf{k_*}$$

## 3. Steps 2 & 3: The Measurement Step (If $z_t$ is available)

If a measurement $z_t$ with base variance $R_t$ is recorded at time $t$, we perform a Bayesian update fusing the GP prediction and the measurement.

**Compute the Innovation:**
$$\nu_t = z_t - \mu_{t|t-1}$$

**Variance Inflation (Innovation Gating):**
Large disagreements between the measurement and the GP prediction inflate the effective measurement variance, reducing trust in outlier-like observations. Let $\lambda > 0$ be a fixed inflation factor. The variance used for the update is:

$$R_t^* = R_t \left(1 + \frac{\nu_t^2}{\lambda}\right)$$

When $\nu_t = 0$, $R_t^* = R_t$ and the update is unchanged. As $|\nu_t|$ grows, $R_t^*$ increases, lowering the Kalman gain and pulling the posterior mean less toward the measurement.

**Compute the Kalman Gain:**
$$K_t = \frac{\Sigma_{t|t-1}}{\Sigma_{t|t-1} + R_t^*}$$

**Update the Mean (Posterior Output):**
$$\mu_{t|t} = \mu_{t|t-1} + K_t \nu_t$$

**Update the Variance (Posterior Variance):**
$$\Sigma_{t|t} = (1 - K_t) \Sigma_{t|t-1}$$

**Data Storage:** Append $t$ to $\mathbf{T}$, $z_t$ to $\mathbf{z}$, and $R_t^*$ to $\mathbf{R_{past}}$. Pop the oldest elements from these buffers to maintain the window length $N$.

## 4. Step 4: The Extrapolation Step (If $z_t$ is NOT available)

If no measurement is recorded at time $t$, the output defaults to the pure GP prediction.

**Output Mean and Variance:**
$$\mu_{t|t} = \mu_{t|t-1}$$
$$\Sigma_{t|t} = \Sigma_{t|t-1}$$

**Data Storage:** **Do not** append anything to the buffers $\mathbf{T}$, $\mathbf{z}$, or $\mathbf{R_{past}}$. By keeping the buffer frozen at the last *measured* time steps, the temporal distance between evaluation time $t$ and the known data increases. This causes the cross-covariance $\mathbf{k}_*$ to shrink, naturally inflating the predicted variance $\Sigma_{t|t-1}$ as the filter extrapolates further into the future.