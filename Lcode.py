from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


data= pd.read_csv("Lcode.csv")
X=data[["MeanTemp","MinTemp"]].values.reshape(-1,2)
y=data["MaxTemp"].values.reshape(-1,1)
print(X)

xtrain,xtest,ytrain,ytest =train_test_split(X,y,test_size=0.2)

lreg=LinearRegression()

lreg.fit(xtrain,ytrain)
print("intercept c is",lreg.intercept_,"coe m is ",lreg.coef_)
ypred=lreg.predict(xtest)
print("actual vrs predicted values for ",xtest)
print(ypred)
print(ytest)
plt.plot(xtest,ypred)
plt.plot(xtest,ytest)
plt.show()
print("MSE",mean_squared_error(ytest,ypred))



'''
y(max temp)=m1x(min temp)+m2x2+m3x3
'''