# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.Shruthi
RegisterNumber:  212222220044
*/
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```
## Output:
Head:
![Screenshot 2024-08-28 083313](https://github.com/user-attachments/assets/85ff1336-7222-4015-ac47-8791bb9ae216)
Tail:
![Screenshot 2024-08-28 083328](https://github.com/user-attachments/assets/ba765eb1-e929-4daf-b48c-09fc41281943)
Array value of x:
![Screenshot 2024-08-28 083351](https://github.com/user-attachments/assets/1c0f12fa-eb92-402d-bac1-41decb2d9c17)
Array value of y:
![Screenshot 2024-08-28 083411](https://github.com/user-attachments/assets/aa5cadfb-3665-4dce-983f-99a0d4f2d542)
Y prediction:
![Screenshot 2024-08-28 083422](https://github.com/user-attachments/assets/0d6aed8d-ab37-48ad-85dd-e1aa632e0bc0)
Array value of Y test:
![Screenshot 2024-08-28 083516](https://github.com/user-attachments/assets/dfe86769-a004-4c67-bed3-f72236a2f191)
Training set graph:
![Screenshot 2024-08-28 083616](https://github.com/user-attachments/assets/a61bb062-e28c-44a7-991e-3b7044a6ac8e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
