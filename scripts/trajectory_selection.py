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

ids_p=[35,113,228,664,246,171,18,86]
ids_v=[65,49,85,92,113,119,9,34]

interaction_p={}
interaction_p['1_04 13 00']=ids_p[0]
interaction_p['1_06 13 00']=ids_p[1]
interaction_p['1_06 37 00']=ids_p[2]
interaction_p['1_07 01 00']=ids_p[3]
interaction_p['1_07 25 00']=ids_p[4]
interaction_p['1_07 49 00']=ids_p[5]
interaction_p['5_04:37:00']=ids_p[6]
interaction_p['5_04:57:00']=ids_p[7]


interaction_v={}
interaction_v['1_04_13_00']=ids_v[0]
interaction_v['1_06_13_00']=ids_v[1]
interaction_v['1_06_37_00']=ids_v[2]
interaction_v['1_07_01_00']=ids_v[3]
interaction_v['1_07_25_00']=ids_v[4]
interaction_v['1_07_49_00']=ids_v[5]
interaction_v['5_04:37:00']=ids_v[6]
interaction_v['5_04:57:00']=ids_v[7]

cam_list=[1,5]
'''
path_pedestrian="../data/trajectories_filtered/p1/"
path_vehicle="../data/trajectories_filtered/v1/"
names_ped=os.listdir(path_pedestrian)
names_veh=os.listdir(path_vehicle)
frames_v={}
for j in cam_list:
    path_pedestrian="../data/trajectories_filtered/p"+str(j)+"/"
    path_vehicle="../data/trajectories_filtered/v"+str(j)+"/"
    names_ped=os.listdir(path_pedestrian)
    names_veh=os.listdir(path_vehicle)
    #print(path_pedestrian)
    #print(names_ped)
    #print(names_ped[0])
    #print(get_num_from_name(names_ped[0]))

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
            id=interaction_v[vid_name[:-22]]
            frames_v[vid_name[:-22]]=ordered_tracks[str(id)][:,0]
            print(vid_name[:-22],len(frames_v[vid_name[:-22]]))
            interaction_v[vid_name[:-22]]=ordered_tracks[str(interaction_v[vid_name[:-22]])][:,1:3]
            #print(vid_name[:-22],len(interaction_v[vid_name[:-22]]))
'''

#for i in frames_v.keys():
#    print(i)
#print(list(frames_v.keys()))
#print(list(interaction_v.keys()))

homograph_1=h_matrix('../tools/Cal_PnP/data/cam_1/calibration.txt')
homograph_3=h_matrix('../tools/Cal_PnP/data/cam_3/calibration.txt')
homograph_5=h_matrix('../tools/Cal_PnP/data/cam_5/calibration.txt')
homograph_13=h_matrix('../tools/Cal_PnP/data/cam_13/calibration.txt')
#print(homograph_1)
#print(homograph_3)
#print(homograph_5)
#print(homograph_13)

frames_v={}
for j in cam_list:
    path_pedestrian="../data/trajectories_filtered/p"+str(j)+"/"
    path_vehicle="../data/trajectories_filtered/v"+str(j)+"/"
    names_ped=os.listdir(path_pedestrian)
    names_veh=os.listdir(path_vehicle)
    #print(path_pedestrian)
    #print(names_ped)
    #print(names_ped[0])
    #print(get_num_from_name(names_ped[0]))

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
            id=interaction_v[vid_name[:-22]]
            frames_v[vid_name[:-22]]=ordered_tracks[str(id)][:,0]
            #print(vid_name[:-22],len(frames_v[vid_name[:-22]]))
            interaction_v[vid_name[:-22]]=ordered_tracks[str(interaction_v[vid_name[:-22]])][:,1:3]
            #print(vid_name[:-22],len(interaction_v[vid_name[:-22]]))

frames_p={}
for j in cam_list:
    path_pedestrian="../data/trajectories_filtered/p"+str(j)+"/"
    path_vehicle="../data/trajectories_filtered/v"+str(j)+"/"
    names_ped=os.listdir(path_pedestrian)
    names_veh=os.listdir(path_vehicle)
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
            id=interaction_p[vid_name[:-22]]
            frames_p[vid_name[:-22]]=ordered_tracks[str(id)][:,0]
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

#print(get_num_from_name())
print(globals()["homograph_"+str(1)])

#print(np.shape(interaction_p['1_04 13 00']))
#inv_homograph=np.linalg.inv(homograph)
interaction_p_temp=interaction_p
interaction_v_temp=interaction_v
interaction_p_gps={}
interaction_v_gps={}
for i in interaction_p_temp.keys():
    #print(np.shape(interaction_p[i]))
    interaction_p_temp[i]=np.append(interaction_p_temp[i],np.ones([len(interaction_p_temp[i]),1]),axis=1)
    #print(np.shape(inv_homograph),np.shape(interaction_p[i]))
    gps=np.matmul(np.linalg.inv(globals()["homograph_"+str(i[0])]),interaction_p_temp[i][:,:3].T)
    gps=gps/gps[2]
    gps=gps.T
    interaction_p_gps[i]=gps[:,:2]

for i in interaction_v_temp.keys():
    #print(np.shape(interaction_v[i]))
    interaction_v_temp[i]=np.append(interaction_v_temp[i],np.ones([len(interaction_v_temp[i]),1]),axis=1)
    #print(np.shape(inv_homograph),np.shape(interaction_v[i]))
    gps=np.matmul(np.linalg.inv(globals()["homograph_"+str(i[0])]),interaction_v_temp[i][:,:3].T)
    gps=gps/gps[2]
    gps=gps.T
    interaction_v_gps[i]=gps[:,:2]

for (k,v), (k2,v2) in zip(interaction_v_gps.items(), interaction_p_gps.items()):
    dfs=pd.DataFrame({"lat" : v[:,0],"lon" : v[:,1]})
    dfs.to_csv("../Results/trajectories_gps/"+k+"_veh"+".csv",index=False)
    dfs=pd.DataFrame({"lat" : v2[:,0],"lon" : v2[:,1]})
    dfs.to_csv("../Results/trajectories_gps/"+k+"_ped"+".csv",index=False)

#print(list(frames_p.keys()))
#print(list(interaction_p_gps.keys()))
#print(list(frames_v.keys()))
#print(list(interaction_v_gps.keys()))

initial_key_p=list(interaction_p_gps.keys())[0]
initial_key_v=list(interaction_v_gps.keys())[0]
dataset_lat_long_p=interaction_p_gps[initial_key_p]
dataset_lat_long_v=interaction_v_gps[initial_key_v]
dataset_frames_p=frames_p[initial_key_p]
dataset_frames_v=frames_v[initial_key_v]
key_list_p=len(interaction_p_gps[initial_key_p])*[initial_key_p]
key_list_v=len(interaction_v_gps[initial_key_v])*[initial_key_v]
id_list_p=len(interaction_p_gps[initial_key_p])*[str(ids_p[0])]
id_list_v=len(interaction_v_gps[initial_key_v])*[str(ids_v[0])]
#print(len(dataset_lat_long_p),len(key_list_p),len(dataset_frames_p))
#print(key_list_p)
count=1
for (k,v), (k2,v2) in zip(interaction_v_gps.items(), interaction_p_gps.items()):
    #print(k,k2,k3,k4)
    if(k!=initial_key_v and k2!=initial_key_p):
        dataset_lat_long_p=np.append(dataset_lat_long_p,v2,axis=0)
        dataset_lat_long_v=np.append(dataset_lat_long_v,v,axis=0)
        dataset_frames_p=np.append(dataset_frames_p,frames_p[k2],axis=0)
        dataset_frames_v=np.append(dataset_frames_v,frames_v[k],axis=0)
        key_list_p=key_list_p+len(v2)*[k2]
        key_list_v=key_list_v+len(v)*[k]
        id_list_p=id_list_p+len(v2)*[str(ids_p[count])]
        id_list_v=id_list_v+len(v)*[str(ids_v[count])]
        count=count+1
        #print(len(dataset_lat_long_p),len(key_list_p),len(dataset_frames_p))

#print(len(dataset_lat_long_p),len(key_list_p),len(dataset_frames_p))
dfs_p=pd.DataFrame({"clip" : key_list_p,"id": id_list_p,"frame":dataset_frames_p,"latitude" : dataset_lat_long_p[:,0],"longitude" : dataset_lat_long_p[:,1] })
dfs_v=pd.DataFrame({"clip" : key_list_v,"id": id_list_v,"frame":dataset_frames_v,"latitude" : dataset_lat_long_v[:,0],"longitude" : dataset_lat_long_v[:,1] })

#dfs=pd.DataFrame({"lat" : dataset_lat_long_p[:,0]})
dfs_p.to_csv("../Results/"+initial_key_p+"_ped"+".csv",index=False)
dfs_v.to_csv("../Results/"+initial_key_v+"_veh"+".csv",index=False)