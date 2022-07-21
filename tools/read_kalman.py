import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
def n_order_dict(ways,point=False):
    num_ids=np.unique(ways[:,0].astype(int))
    ordered_tracks={}
    ordered_tracks_point_class={}
    count=0
    for i in num_ids:
        count=count+1
      #ordered_tracks['id_'+str(i)]=ways[ways[:,0]==(i),1:]
        ordered_tracks[str(i)]=ways[ways[:,0]==(i),1:]
    if(point):
        for j in range(len(ways)):
        #ordered_tracks_point_class['id_'+str(j)]=Point(ways[j,1:])
            ordered_tracks_point_class[str(j)]=Point(ways[j,1:])
        return ordered_tracks_point_class
    return num_ids,ordered_tracks

f_data=pd.read_csv("../data/trajectories_filtered/1/1_12 33 00_traj_ped_filtered.csv")
filt_data=np.zeros([len(f_data),6])
filt_data[:,0]=f_data['id']
filt_data[:,1]=f_data['frame']
filt_data[:,2]=f_data['x_est']
filt_data[:,3]=f_data['y_est']
filt_data[:,4]=f_data['vx_est']
filt_data[:,5]=f_data['vy_est']

n,f_data_dic=n_order_dict(filt_data)
img=cv2.imread('../tools/Cal_PnP/pic/frm.jpg')[..., ::-1]
for i in f_data_dic.keys():
    #print(i)
    #print(np.shape(f_data_dic[i]))
    #if i=="158":
    plt.plot(f_data_dic[i][:,1],f_data_dic[i][:,2],'.',markersize=2)
plt.imshow(img)
plt.savefig('../images/kalman_trajectories.png',dpi=300)
plt.show()
#print(f_data_dic["422"][:,0])

plt.show()
