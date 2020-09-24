import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("longdata.csv")

X = data[["year","age"]]
y = data["group"]

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.4)


log = LogisticRegression()
log.fit(Xtrain,ytrain)
ypred= log.predict(Xtest)
print(accuracy_score(ypred,ytest))

cf=confusion_matrix(ypred,ytest)
print(cf)
