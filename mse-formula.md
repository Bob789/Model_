# Mean Squared Error (MSE)

Mean Squared Error is a common metric used to evaluate the performance of regression models. It measures the average of the squared differences between predicted values and actual values.

## Formula

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $n$ is the number of data points
- $y_i$ is the actual value for the $i$-th observation
- $\hat{y}_i$ is the predicted value for the $i$-th observation

## Interpretation

- MSE is always non-negative, with values closer to 0 indicating better model fit
- The squaring operation penalizes larger errors more heavily than smaller ones
- The units of MSE are the square of the units of the dependent variable
- It gives a higher weight to outliers due to the squaring operation

## Implementation in Python

```python
from sklearn.metrics import mean_squared_error

# Example
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
# Output: Mean Squared Error: 0.3750
```

## Relationship to Other Metrics

- **RMSE (Root Mean Squared Error)**: $\text{RMSE} = \sqrt{\text{MSE}}$
- **MAE (Mean Absolute Error)**: Similar to MSE but uses absolute differences instead of squared differences
- **R² (Coefficient of Determination)**: Related to MSE through the formula $R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)}$
