import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv('diabetes_data.csv')
#create a dataframe with all training data except the target column
x = df.drop(columns=['diabetes'])
#check that the target variable has been removed
print(x.head())
#Separate target values
y = df['diabetes'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(x, y, test_size = 0.2 ,random_state=10)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier (n_neighbors = 5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print("confusion matrix\n",confusion_matrix(y_pred,y_test))
print("accuracy score", accuracy_score(y_pred,y_test))

x1 = df.drop(columns=['diabetes'])
print(x.head())

y1 = df['diabetes'].values
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test =train_test_split(x1, y1, test_size = 0.2 ,random_state=10)

kmeans =KMeans(n_clusters=2)
kmeans.fit(X1_train,y1_train)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_

y1_pred =kmeans.predict(X1_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print("confusion matrix\n",confusion_matrix(y1_pred,y1_test))
print("accuracy score", accuracy_score(y1_pred,y1_test))

acc_data = {"KNN":0,"KMEAN":0}
acc_data["KNN"]=accuracy_score(y_pred,y_test)
acc_data["KMEAN"]=accuracy_score(y1_pred,y1_test)
print (acc_data)

plt.bar(acc_data.keys(),acc_data.values())
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Comparative Study")
plt.show()



