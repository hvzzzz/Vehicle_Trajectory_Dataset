import geopandas as gpd
from geopandas import GeoDataFrame
import sys
sys.path.append("/home/han4n/Vehicle_Trayectory_Dataset/")
from tools.homograph_matrix_read import h_matrix
from tools.utils import *
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pickle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy import signal
from scipy import ndimage
from scipy import interpolate

interaction_p={}
interaction_p['1_04 13 00']=35
interaction_p['1_06 13 00']=113
interaction_p['1_06 37 00']=228
interaction_p['1_07 01 00']=664
interaction_p['1_07 25 00']=246
interaction_p['1_07 49 00']=171

interaction_v={}
interaction_v['1_04_13_00']=65
interaction_v['1_06_13_00']=49
interaction_v['1_06_37_00']=85
interaction_v['1_07_01_00']=92
interaction_v['1_07_25_00']=113
interaction_v['1_07_49_00']=119

path_pedestrian="../data/trajectories_filtered/p1/"
path_vehicle="../data/trajectories_filtered/v1/"
names_ped=os.listdir(path_pedestrian)
names_veh=os.listdir(path_vehicle)

homograph=h_matrix('../tools/Cal_PnP/data/calibration.txt')
homograph

for vid_name in names_veh:
    if(vid_name[-3:]=="csv"):
        print("Processing Vehicles "+vid_name)
        data_df=pd.read_csv(path_vehicle+vid_name,sep=",")
        data=np.zeros([len(data_df),6])
        data[:,0]=data_df['id'].to_numpy('float')
        data[:,1]=data_df['frame'].to_numpy('float')
        data[:,2]=data_df['x_est'].to_numpy('float')
        data[:,3]=data_df['y_est'].to_numpy('float')
        data[:,4]=data_df['vx_est'].to_numpy('float')
        data[:,5]=data_df['vy_est'].to_numpy('float')
        num_ids,ordered_tracks=n_order_dict(data)
        interaction_v[vid_name[:-22]]=ordered_tracks[str(interaction_v[vid_name[:-22]])][:,1:3]

for vid_name in names_ped:
    if(vid_name[-3:]=="csv"):
        print("Processing Vehicles "+vid_name)
        data_df=pd.read_csv(path_pedestrian+vid_name,sep=",")
        data=np.zeros([len(data_df),6])
        data[:,0]=data_df['id'].to_numpy('float')
        data[:,1]=data_df['frame'].to_numpy('float')
        data[:,2]=data_df['x_est'].to_numpy('float')
        data[:,3]=data_df['y_est'].to_numpy('float')
        data[:,4]=data_df['vx_est'].to_numpy('float')
        data[:,5]=data_df['vy_est'].to_numpy('float')
        num_ids,ordered_tracks=n_order_dict(data)
        interaction_p[vid_name[:-22]]=ordered_tracks[str(interaction_p[vid_name[:-22]])][:,1:3]

img=cv2.imread('../tools/Cal_PnP/pic/frm.jpg')[..., ::-1]
for (k,v), (k2,v2) in zip(interaction_v.items(), interaction_p.items()):
    plt.plot(v[:,0],v[:,1],label='Vehicle')
    plt.plot(v2[:,0],v2[:,1],label='Pedestrian')
    plt.title(k+" "+k2)
    plt.legend()
    fname='../images/'+k+'_'+k2+'.png'
    plt.imshow(img)
    #plt.savefig(fname)
    #plt.show()

#print(np.shape(interaction_p['1_04 13 00']))
inv_homograph=np.linalg.inv(homograph)
interaction_p_temp=interaction_p
interaction_v_temp=interaction_v
interaction_p_gps={}
interaction_v_gps={}
for i in interaction_p_temp.keys():
    #print(np.shape(interaction_p[i]))
    interaction_p_temp[i]=np.append(interaction_p_temp[i],np.ones([len(interaction_p_temp[i]),1]),axis=1)
    #print(np.shape(inv_homograph),np.shape(interaction_p[i]))
    gps=np.matmul(inv_homograph,interaction_p_temp[i][:,:3].T)
    gps=gps/gps[2]
    gps=gps.T
    interaction_p_gps[i]=gps[:,:2]

for i in interaction_v_temp.keys():
    #print(np.shape(interaction_v[i]))
    interaction_v_temp[i]=np.append(interaction_v_temp[i],np.ones([len(interaction_v_temp[i]),1]),axis=1)
    #print(np.shape(inv_homograph),np.shape(interaction_v[i]))
    gps=np.matmul(inv_homograph,interaction_v_temp[i][:,:3].T)
    gps=gps/gps[2]
    gps=gps.T
    interaction_v_gps[i]=gps[:,:2]

for (k,v), (k2,v2) in zip(interaction_v_gps.items(), interaction_p_gps.items()):
    dfs=pd.DataFrame({"lat" : v[:,0],"lon" : v[:,1]})
    dfs.to_csv("../Results/trajectories_gps/"+k+"_veh"+".csv",index=False)
    dfs=pd.DataFrame({"lat" : v2[:,0],"lon" : v2[:,1]})
    dfs.to_csv("../Results/trajectories_gps/"+k+"_ped"+".csv",index=False)
