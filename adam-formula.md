# Adam (Adaptive Moment Estimation) Update Formula

## Algorithm Steps:

1. **Initialize** parameters and hyperparameters:
   - $\theta_0$: Initial parameter vector
   - $m_0 = 0$: First moment vector (momentum)
   - $v_0 = 0$: Second moment vector (velocity)
   - $\alpha$: Learning rate (typically 0.001)
   - $\beta_1$: Exponential decay rate for first moment (typically 0.9)
   - $\beta_2$: Exponential decay rate for second moment (typically 0.999)
   - $\epsilon$: Small constant to prevent division by zero (typically 1e-8)

2. **For** each time step $t = 1, 2, ...$:
   
   a. Compute gradient: $g_t = \nabla_\theta f_t(\theta_{t-1})$
   
   b. Update biased first moment estimate:
   $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
   
   c. Update biased second moment estimate:
   $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
   
   d. Correct bias in first moment:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
   
   e. Correct bias in second moment:
   $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   
   f. Update parameters:
   $$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## Compact Form:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla_\theta f_t(\theta_{t-1})$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot \nabla_\theta f_t(\theta_{t-1})^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$