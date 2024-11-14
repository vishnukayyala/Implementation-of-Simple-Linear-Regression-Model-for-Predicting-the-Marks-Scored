

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: pandas, numpy, matplotlib, and scikit-learn.
2. Load the dataset `student_scores.csv` into a DataFrame and print it to verify contents.
3. Display the first and last few rows of the DataFrame to inspect the data structure.
4. Extract the independent variable (`x`) and dependent variable (`y`) as arrays from the DataFrame.
5. Split the data into training and testing sets, with one-third used for testing and a fixed `random_state` for reproducibility.
6. Create and train a linear regression model using the training data.
7. Make predictions on the test data and print both the predicted and actual values for comparison.
8. Plot the training data as a scatter plot and overlay the fitted regression line to visualize the model's fit.
9. Plot the test data as a scatter plot with the regression line to show model performance on unseen data.
10. Calculate and print error metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for evaluating model accuracy.
11. Display the plots to visually assess the regression results.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Vishnu KM
RegisterNumber:212223240185
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:


![image](https://github.com/user-attachments/assets/6182a7e7-af97-4035-b10a-c88d3dce3052)

![image](https://github.com/user-attachments/assets/96845ce2-a3d3-47c6-921c-27f191e8aad1)

![image](https://github.com/user-attachments/assets/d8cb806c-e9e3-4163-8141-2ad05571b7ba)

![image](https://github.com/user-attachments/assets/93dc9902-52a2-4604-90a6-2c195e4a65c5)


![Screenshot 2024-08-29 114303](https://github.com/user-attachments/assets/3fe4d593-4b2a-4cc7-80d9-696ce3da86b5)

![image](https://github.com/user-attachments/assets/de4f8774-a598-44aa-a75a-1bc66257eb40)


![image](https://github.com/user-attachments/assets/7f2934a4-718b-423d-8cc7-22942fcf6a94)

![image](https://github.com/user-attachments/assets/0cbbce4d-4e4b-4a16-aa2b-ce9e987748ec)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
