import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#df_ped=pd.read_csv("../data/trajectories/1/1_12 33 00/p158.csv")
df_ped=pd.read_csv("../../vci-dataset-citr/data/trajectories/p2p_bi/bidirection_no_vehicle_3v7_01/p1.csv")
plt.plot(df_ped.x,df_ped.y)
id_frames=list(df_ped.frame)
#print(list(df_ped["frame"]))
xy=np.zeros([len(df_ped),2])
xy[:,0]=df_ped['x'].to_numpy('float')
xy[:,1]=df_ped['y'].to_numpy('float')
print(len(xy[:,0]))

for i, id_frame in enumerate(id_frames):
    id_dataframe = int(df_ped[df_ped.frame == id_frame].index.values)
    #print(i,len(id_frames))
    #print(df_ped.frame == id_frame + 1)
    #print(df_ped.frame)
    #print(id_frame+1)
    #print(df_ped[df_ped.frame == id_frame + 1].x)
    #print(df_ped[df_ped.frame == id_frame + 1].y)
    #print(float(df_ped[df_ped.frame == id_frame + 1].x),float(df_ped[df_ped.frame == id_frame + 1].y))
    #input('')
    if i< len(id_frames)-1:
        print(i,len(id_frames))
        #z_new = np.array([[float(df_ped[df_ped.frame == id_frame + 1].x)],
                          #[float(df_ped[df_ped.frame == id_frame + 1].y)]])
        z_new = np.array([xy[:,0][i+1],xy[:,1][i+1]])
        print(z_new)
        print(float(df_ped[df_ped.frame == id_frame + 1].x),float(df_ped[df_ped.frame == id_frame + 1].y))
 
