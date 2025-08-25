# Regression Exercise — Discover the Relationship 🏋️‍♂️📊

**Goal**: Use a scatter plot to visually guess the type of relationship between the variables, then choose the appropriate regression model

## Dataset 1

This dataset represents the relationship between **hours spent in the gym per week** (X) and **muscle mass gain in kg** (Y)

| x (hours/week) | y (kg muscle gain) |
| -------------: | -----------------: |
|              0 |                3.1 |
|              1 |                5.0 |
|              2 |                7.2 |
|              3 |                8.9 |
|              4 |               11.2 |
|              5 |               12.8 |
|              6 |               15.2 |
|              7 |               17.1 |
|              8 |               19.2 |
|              9 |               21.0 |

## Dataset 2

This dataset represents the relationship between **daily weightlifting load in kg** (X) and **muscle strength score** (Y)

| x (load kg) | y (strength score) |
| ----------: | -----------------: |
|          -5 |              -20.5 |
|          -4 |              -15.2 |
|          -3 |               -7.6 |
|          -2 |               -0.8 |
|          -1 |                3.2 |
|           0 |                2.1 |
|           1 |                4.7 |
|           2 |                6.0 |
|           3 |                4.9 |
|           4 |                1.6 |
|           5 |               -3.8 |

---

## Your tasks

1. 📈 Plot the data (scatter plot) for each dataset and **decide visually** what kind of pattern it shows
2. 🔍 If the pattern looks linear, fit **Linear Regression**; if it looks curved, fit **Polynomial Regression** (degree=2)
3. 📊 **For Polynomial Regression only**: Split the dataset into **train/test** (80/20, random\_state=42) and fit the model on train only
4. 📉 **For Polynomial Regression only**: Predict for both train and test sets
5. 🧮 **For Polynomial Regression only**: Report **MSE, MAE, RMSE, R², Adjusted R²** for train and test
6. 🎯 For curved models, find the **optimal point (vertex)** using $x^* = -\frac{b}{2a}$ and plot it
7. 📏 **For Linear Regression only**: Calculate and report **MSE, MAE, RMSE, R²** for the dataset

💡 **Remark**: For Polynomial Regression you can use either the `LinearRegression` class from scikit-learn with `PolynomialFeatures` or the `np.polyfit` function from NumPy.

**Bonus**: Try polynomial degrees 1–4, compare test Adjusted R², and see which one generalizes better


## 📤 הגשה

יש לשלוח את הפתרון למייל:
📧 [pythonai250824+linpolhw@gmail.com](mailto:pythonai250824+linpolhw@gmail.com)
