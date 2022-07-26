from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
f_data=pd.read_csv("../data/trajectories_filtered/1/1_12 33 00_traj_ped_filtered.csv")
X=np.zeros([len(f_data),3])
X[:,0]=f_data['x_est']
X[:,1]=f_data['y_est']
X[:,2]=f_data['frame']
#X = StandardScaler().fit_transform(X)
#X = np.array([[1, 2], [2, 2], [2, 3],
#              [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
plt.scatter(X[:,0], X[:,1])
plt.show()

print(clustering.labels_)
print(clustering)
