import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data=pd.read_csv("../data/trajectories_filtered/1/1_12 33 00_traj_ped_filtered.csv")
X=np.zeros([len(data),3])
X[:,0]=data['x_est']
X[:,1]=data['y_est']
X[:,2]=data['frame']
#print(data.head())
cols = list(data.columns.values)
cols.pop(cols.index('x_est'))
data = data[cols+['x_est']]
cols = list(data.columns.values)
cols.pop(cols.index('y_est'))
data = data[cols+['y_est']]
cols = list(data.columns.values)
cols.pop(cols.index('frame'))
data = data[cols+['frame']]
#X = data.iloc[:,7:10].values
#elbow method
#wcss = []
#for i in range(1,12):
#    k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
#    k_means.fit(X)
#    wcss.append(k_means.inertia_)#plot elbow curve
#plt.plot(np.arange(1,12),wcss)
#plt.xlabel('Clusters')
#plt.ylabel('SSE')
#plt.show()
k_means_optimum = KMeans(n_clusters = 2, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(X)
print(y)
data['cluster'] = y  
data1 = data[data.cluster==0]
data2 = data[data.cluster==1]
kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)
kplot.plot3D(xline, yline, zline, 'black')# Data for three-dimensional scattered points
kplot.scatter3D(data1.x_est, data1.y_est, data1.frame, c='red', label = 'Cluster 1',s=0.1)
kplot.scatter3D(data2.x_est,data2.y_est,data2.frame,c ='green', label = 'Cluster 2',s=0.1)
plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 200)
plt.legend()
plt.title("Kmeans")
from sklearn.metrics import silhouette_score
score = silhouette_score(X,y)
print(score)
plt.show()
