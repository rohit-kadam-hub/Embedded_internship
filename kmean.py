import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.style as style
import pandas as pd

style.use("ggplot")
data = pd.read_csv("loan.csv")
new_data = data[["salary ", "EMI"]]
X = np.array(new_data)
print(X)

kmeans = KMeans(n_clusters=2)
y = kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)
colors = ["g.", "r.", "c.", "y."]

xtest = input("Enter the test x value")
ytest = input("enter the test y value")
value = kmeans.predict([[xtest, ytest]])
print("pt lies @", value[0], "cluster")
for i in range(len(X)):
    print("coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x")
plt.plot(xtest, ytest, colors[value[0]], markersize=10)
plt.show()
