from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

#X = np.array([[1, 2], [1, 4], [1, 0],
#              [10, 2], [10, 4], [10, 0]])
#print(np.shape(X))

f_data=pd.read_csv("../data/trajectories_filtered/1/1_12 33 00_traj_ped_filtered.csv")
X=np.zeros([len(f_data),2])
X[:,0]=f_data['x_est']
X[:,1]=f_data['y_est']

kmeans = KMeans(n_clusters=17, random_state=0).fit(X)
print(kmeans.labels_)
kmeans.predict([[0, 0], [12, 3]])

kmeans.cluster_centers_

