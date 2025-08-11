# רגרסיה לינארית מרובת משתנים

## הבעיה שרוצים לפתור

רגרסיה לינארית מרובת משתנים (Multiple Linear Regression) מרחיבה את מודל הרגרסיה הלינארית הפשוטה ומאפשרת לנו לבחון את הקשר בין משתנה תלוי אחד ומספר משתנים בלתי תלויים. כך ניתן לבנות מודלים מורכבים יותר ולשפר את יכולת החיזוי.

**דוגמה**: נניח שאנו רוצים לחזות את מחיר הדירה בהתבסס על מספר מאפיינים. בניגוד לרגרסיה לינארית פשוטה, שעשויה להתייחס רק לשטח הדירה, רגרסיה מרובת משתנים מאפשרת לנו לשקלל גם גורמים נוספים כמו מספר חדרים, גיל המבנה, מיקום, וכדומה.

להלן דוגמה לנתונים של דירות:

| שטח (מ"ר) | מספר חדרים | גיל המבנה (שנים) | מרחק ממרכז העיר (ק"מ) | מחיר (אלפי ש"ח) |
|-----------|------------|-----------------|----------------------|-----------------|
| 70        | 3          | 15              | 5                    | 1,200           |
| 90        | 4          | 10              | 7                    | 1,500           |
| 60        | 2          | 20              | 3                    | 1,100           |
| 120       | 5          | 5               | 10                   | 1,800           |
| 80        | 3          | 12              | 6                    | 1,300           |
| 110       | 4          | 8               | 8                    | 1,650           |
| 100       | 4          | 7               | 5                    | 1,750           |
| 75        | 3          | 18              | 4                    | 1,250           |
| 95        | 4          | 9               | 6                    | 1,550           |
| 130       | 5          | 3               | 12                   | 1,900           |

המטרה שלנו היא לבנות מודל שיאפשר לנו לחזות את מחיר הדירה בהתבסס על המאפיינים השונים, ולהבין את ההשפעה היחסית של כל מאפיין על המחיר.

## Mathematical Formula and Complete Calculation

Multiple linear regression extends the simple linear regression model to include multiple predictor variables. The general form of the equation is:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \varepsilon$$

Where:
- $y$ is the dependent variable (in our case: apartment price)
- $x_1, x_2, ..., x_p$ are the independent variables (in our case: area, number of rooms, age, distance)
- $\beta_0, \beta_1, \beta_2, ..., \beta_p$ are the coefficients
- $\varepsilon$ is the error term

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Example features and target
X = np.array([
    [70, 3, 15, 5],
    [90, 4, 10, 7],
    [60, 2, 20, 3],
    [120, 5, 5, 10],
    [80, 3, 12, 6],
    [110, 4, 8, 8],
    [100, 4, 7, 5],
    [75, 3, 18, 4],
    [95, 4, 9, 6],
    [130, 5, 3, 12]
])
y = np.array([1200, 1500, 1100, 1800, 1300, 1650, 1750, 1250, 1550, 1900])

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Coefficients (β₁, β₂, β₃, β₄): {model.coef_}")
```

Using the example data, we get the following coefficients:
- $\beta_0 \approx 740.66$ (Intercept)
- $\beta_1 \approx 11.31$ (Area coefficient)
- $\beta_2 \approx 40.11$ (Rooms coefficient)
- $\beta_3 \approx -15.69$ (Building age coefficient)
- $\beta_4 \approx -41.38$ (Distance coefficient)

Therefore, our multiple linear regression equation is:
$$\text{Price} = 740.66 + 11.31 \times \text{Area} + 40.11 \times \text{Rooms} - 15.69 \times \text{Age} - 41.38 \times \text{Distance}$$

This means:
- Each additional square meter increases the price by about 11,310 ILS
- Each additional room increases the price by about 40,110 ILS
- Each year of building age decreases the price by about 15,690 ILS
- Each kilometer of distance from the city center decreases the price by about 41,380 ILS (which aligns with the common pattern that properties closer to city centers typically command higher prices)


## **תרגיל**: 
חברת כוח אדם אספה נתונים על משכורות עובדים בתחום ההייטק. הם מעוניינים לבנות מודל שיחזה את המשכורת בהתבסס על מספר מאפיינים. הנה הנתונים:

| שנות ניסיון | השכלה (שנים) | שעות עבודה שבועיות | נסיון ניהולי (שנים) | משכורת חודשית (אלפי ש"ח) |
|--------------|--------------|---------------------|----------------------|----------------------------|
| 2            | 15           | 40                  | 0                    | 15                         |
| 5            | 16           | 45                  | 1                    | 25                         |
| 3            | 16           | 40                  | 0                    | 18                         |
| 10           | 18           | 50                  | 5                    | 45                         |
| 7            | 17           | 45                  | 3                    | 35                         |
| 1            | 14           | 35                  | 0                    | 12                         |
| 8            | 16           | 45                  | 4                    | 38                         |
| 4            | 15           | 40                  | 1                    | 22                         |
| 6            | 15           | 42                  | 2                    | 30                         |
| 12           | 19           | 55                  | 8                    | 60                         |

משימות:
1. חשב את המקדמים של מודל רגרסיה לינארית מרובת משתנים עבור נתונים אלה
2. כתוב את משוואת החיזוי המתקבלת
3. חזה את המשכורת הצפויה לעובד עם 6 שנות ניסיון, 16 שנות השכלה, 43 שעות עבודה שבועיות, ו-2 שנות ניסיון ניהולי
4. אילו מהמשתנים הם המשפיעים ביותר על המשכורת? הסבר כיצד הגעת למסקנה זו

## 📤 הגשה

יש לשלוח את הפתרון למייל:
📧 [pythonai250824+linmvhw@gmail.com](mailto:pythonai250824+linmvhw@gmail.com)
