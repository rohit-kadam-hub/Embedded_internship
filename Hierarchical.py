import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
customer_data = pd.read_csv ('shopping-data-1.csv')
data = customer_data.iloc[:, 3:5]. values
print (data)
import scipy.cluster.hierarchy as shc
plt.figure (figsize=(10, 7))
plt.title ("Customer Dendograms")
dend = shc.dendrogram(shc.linkage (data, method='ward'))
cluster = AgglomerativeClustering (n_clusters=4, affinity='euclidean')
cluster.fit_predict (data)
plt.figure (figsize= (10, 7))
plt.scatter (data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
print (cluster. labels_)
#DBSCAN Algorithm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
customer_data = pd. read_csv('shopping-data-1.csv')
data= customer_data.iloc[:, 3:5].values
scaler = StandardScaler ()
X_scaled = scaler.fit_transform (data)
dbscan  = DBSCAN (eps=0.2, min_samples = 3)
clusters = dbscan.fit_predict (X_scaled)

#plot the cluster assignments
plt.scatter (data[:, 0], data[:, 1], c=clusters, cmap="plasma")
plt.show()