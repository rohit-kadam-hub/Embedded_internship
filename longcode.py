#nb=
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

acc_data={"DCT_ACC":0,"LOG_ACC":0,"RFC_ACC":0,"SVC_ACC":0}

data = pd.read_csv("longdata.csv")

X = data[["year","age"]]
y = data["group"]

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)

log = LogisticRegression()
log.fit(Xtrain,ytrain)
ypred_log= log.predict(Xtest)

dct = DecisionTreeClassifier()
dct.fit(Xtrain,ytrain)
ypred_dct=dct.predict(Xtest)

rfc=RandomForestClassifier(n_estimators=500)
rfc.fit(Xtrain,ytrain)
ypred_rfc=rfc.predict(Xtest)

svc1=SVC(kernel="rbf")
svc1.fit(Xtrain,ytrain)
ypred_svc=svc1.predict(Xtest)

acc_data["DCT_ACC"]=accuracy_score(ytest,ypred_dct)
acc_data["LOG_ACC"]=accuracy_score(ytest,ypred_log)
acc_data["RFC_ACC"]=accuracy_score(ytest,ypred_rfc)
acc_data["SVC_ACC"]=accuracy_score(ytest,ypred_svc)
print (acc_data)

plt.bar(acc_data.keys(),acc_data.values())
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Comparative Study")
plt.show()





















'''
accvalue=accuracy_score(ypred,ytest)

print ("your prediction accuracy is:",accvalue)
print ("c matrx is:")
from sklearn.metrics import confusion_matrix
print (confusion_matrix(ytest,ypred))
ytests=list(ytest)
correctvalue=0
wrongvalues=0
for i in range(len(ytests)):
    if(ytests[i]==ypred[i]):
        correctvalue=correctvalue+1
    else:
        wrongvalues=wrongvalues+1

import matplotlib.pyplot as plt
plt.bar(["correct","wrong"],[correctvalue,wrongvalues])
plt.show()
'''




