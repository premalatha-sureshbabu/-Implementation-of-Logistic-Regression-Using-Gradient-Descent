# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and Load the dataset.

2.Define X and Y array and Define a function for costFunction,cost and gradient.

3.Define a function to plot the decision boundary.

4.Define a function to predict the Regression value.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S.Prema Latha
RegisterNumber: 212222230112

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
## Array value of x:
![Screenshot 2023-09-23 113924](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/7e48ad70-907e-4585-a806-fcead19a2d76)

## Array value of y:
![Screenshot 2023-09-23 113930](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/a7e1667e-8d94-4909-be48-ebeebaf6ac72)

## score graph:
![Screenshot 2023-09-23 113945](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/b34a3003-38bf-40fc-870f-5c368ecf4982)

## sigmoid function graph:
![Screenshot 2023-09-23 114000](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/98cc6cc3-da72-44bf-9ab0-b8c667afe50a)

## x train grad value:
![Screenshot 2023-09-23 114012](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/41cab977-c53f-4e14-b70f-3cf2f1ed406f)

## y train grad value:
![Screenshot 2023-09-23 114019](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/7e15370a-ad0c-4817-9470-1251145c316b)

## regression value:
![Screenshot 2023-09-23 114031](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/dff70077-50b7-4310-8244-b179f3eaecef)

## decision boundary graph:
![Screenshot 2023-09-23 114059](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/37ef683a-bf67-4590-863d-8178891be739)

## Probablity value:
![Screenshot 2023-09-23 114109](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/e2066e39-2e84-40db-b6c5-bd5468b7a825)

## Prediction value of mean:
![Screenshot 2023-09-23 114114](https://github.com/premalatha-sureshbabu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120620842/42e715be-3ac1-4122-9c87-a21100189de1)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

