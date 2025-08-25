# Logistic Regression — Homework (liblinear)

## Exercise 1 — Binary Logistic Regression (single feature)

**Story**: *x* is the **minutes of brisk activity per day** and *y* is whether the person **hit 10k steps** that day (1 = yes, 0 = no)

### Dataset

| x (minutes) | y (hit 10k steps) |
| ----------: | ----------------: |
|           5 |                 0 |
|          12 |                 0 |
|          18 |                 0 |
|          22 |                 0 |
|          28 |                 0 |
|          35 |                 0 |
|          42 |                 0 |
|          48 |                 1 |
|          55 |                 1 |
|          63 |                 1 |
|          72 |                 1 |
|          85 |                 1 |

### Your tasks

1. Fit a logistic regression model using scikit-learn with `solver='liblinear'` on the table above
2. Print `intercept_` and `coef_`
3. **Decision boundary for 70%**: solve for $x$ such that $p(y=1\mid x) = 0.70$   
5. Compute predicted probabilities for `x = [16, 27, 33, 49, 67, 90]` and list them

### Bonus — Model Evaluation

1. Split the dataset into **train/test** sets (e.g. 70%/30%)
2. Fit the logistic regression model on the training set
3. Use the test set to produce predictions
4. Compute and display the following metrics:

   * Confusion Matrix
   * Accuracy
   * Precision
   * Recall
   * F1-score


## Exercise 2 — Logistic Regression with Two Features (multi-binary)

**Story**: *x1* is **study hours per week** and *x2* is **average sleep hours per night**. *y* is **passed the mock exam** (1 = pass, 0 = fail)

### Dataset

| x1 (study h/wk) | x2 (sleep h/night) | y (pass) |
| --------------: | -----------------: | -------: |
|               2 |                5.0 |        0 |
|               3 |                5.5 |        0 |
|               4 |                5.0 |        0 |
|               4 |                6.0 |        0 |
|               5 |                5.5 |        0 |
|               5 |                6.5 |        0 |
|               6 |                6.0 |        1 |
|               6 |                7.0 |        1 |
|               7 |                6.5 |        1 |
|               7 |                7.5 |        1 |
|               8 |                6.0 |        1 |
|               8 |                7.0 |        1 |
|               9 |                7.0 |        1 |
|              10 |                7.5 |        1 |

### Your tasks

1. Fit a logistic regression with `solver='liblinear'` using features `[x1, x2]`
2. Print `intercept_` and all entries of `coef_` (there should be one row with two values: `[β1, β2]`)
3. *Bonus: Write the regression equation
4. Compute the predicted probability for the point `[x1, x2] = [6.5, 7.5]`

**Submission email**: [pythonai250824+loghw@gmail.com](mailto:pythonai250824+loghw@gmail.com)
