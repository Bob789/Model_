# 🌟 רגרסיה לוגיסטית מולטיוואריאטית לפלט בינארי (Binary Logistic Regression)
  
## 📘 מה זו רגרסיה לוגיסטית בינארית?
רגרסיה לוגיסטית בינארית היא שיטה סטטיסטית המיועדת לחיזוי משתנה תלוי שיש לו בדיוק **שתי אפשרויות בלבד** (למשל: הצלחה/כישלון, כן/לא, חיובי/שלילי), תוך שימוש בכמה משתנים מסבירים (features).
  
 
## ✏️ מושגים בסיסיים:
- **משתנה תלוי (Y)**: משתנה בינארי שאנו מנסים לחזות (למשל: עבר/נכשל במבחן).
- **משתנים בלתי תלויים (X₁, X₂, X₃, ...)**: התכונות המסבירות את Y.
- **β (בטא)**: מקדמים שנלמדים על ידי המודל. כל משתנה מקבל β משלו.
  
 
## 📐 נוסחה מתמטית:
ההסתברות שהמשתנה Y יהיה 1, בהתחשב ב־n משתנים מסבירים:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?P(y%20=%201%20\mid%20x)%20=%20\frac{1}{1%20+%20e^{-(\beta_0%20+%20\beta_1%20x_1%20+%20\beta_2%20x_2%20+%20\dots%20+%20\beta_n%20x_n)}}"/></p>  
  
  
- <img src="https://latex.codecogs.com/gif.latex?\beta_0"/>: intercept (היסט)
- <img src="https://latex.codecogs.com/gif.latex?\beta_1,%20\dots,%20\beta_n"/>: מקדמים של התכונות
- <img src="https://latex.codecogs.com/gif.latex?x_1,%20\dots,%20x_n"/>: ערכי התכונות
- <img src="https://latex.codecogs.com/gif.latex?e"/>: בסיס הלוגריתם הטבעי (≈ 2.718)
  
הנוסחה מחזירה הסתברות בין 0 ל־1.
  
 
## 💬 דוגמה מילולית:
  
נניח שאנו רוצים לחזות אם סטודנט יעבור מבחן (כן/לא) לפי שעות למידה (X₁) ומספר תרגילים שנפתרו (X₂):
  
למשל:
  
- סטודנט שלמד 10 שעות ופתר 15 תרגילים
- נניח שהמודל חישב את הציון (logit):
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?z%20=%20\beta_0%20+%20\beta_1%20\cdot%2010%20+%20\beta_2%20\cdot%2015"/></p>  
  
  
נניח:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?z%20=%20-1%20+%200.2%20\cdot%2010%20+%200.1%20\cdot%2015%20=%201.5"/></p>  
  
  
ההסתברות שהסטודנט יעבור:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?P%20=%20\frac{1}{1%20+%20e^{-1.5}}%20\approx%200.8176"/></p>  
  
  
כלומר, הסתברות של כ־81.8% שהסטודנט יעבור.
  
## 🔍 פירוש המקדמים (β):
- מקדם חיובי (β>0) מציין שככל שהתכונה גדלה, עולה ההסתברות לתוצאה חיובית (y=1).
- מקדם שלילי (β<0) מציין שככל שהתכונה גדלה, ההסתברות לתוצאה חיובית (y=1) יורדת.
- מקדם קרוב ל־0 מציין שהתכונה אינה משפיעה משמעותית על התוצאה.
   
## 🧪 דוגמת קוד בפייתון:
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Data: [Age, Income]
X = np.array([
    [25, 30],   # young, low income
    [30, 35],
    [35, 40],
    [40, 60],   # middle age, medium income
    [45, 65],
    [50, 70],
    [55, 80],   # older, higher income
    [60, 85]
])

# Target: Loan approval (1=yes, 0=no)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Fit logistic regression model
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# Get coefficients
b0 = model.intercept_[0]
b1, b2 = model.coef_[0]

# Print the logistic regression equation
print("Logistic regression equation:")
print(f"P(loan=1) = 1 / (1 + e^-({b0:.2f} + {b1:.2f}*Age + {b2:.2f}*Income))")

# Predict probability for a new person
new_person = np.array([[40, 50]])  # Age=40, Income=50
prob = model.predict_proba(new_person)[0, 1]
print(f"Predicted probability of getting a loan: {prob:.2f}")

```
  