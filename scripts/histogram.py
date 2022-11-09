from geopy.distance import geodesic
import pandas as pd
import matplotlib.pyplot as plt

path_res_ped="../Results/1_06_37_00_ped.csv"
path_res_veh="../Results/1_06_37_00_veh.csv"
traj_ped=pd.read_csv(path_res_ped,sep=",")
traj_veh=pd.read_csv(path_res_veh,sep=",")
fps=30
delta_time=1/fps
speed_ped=[];speed_veh=[]
for i in range(len(traj_ped)):
    if(i==len(traj_ped)-1):
        break
    if(traj_ped['id'][i]==traj_ped['id'][i+1]):
        speed=geodesic([traj_ped['latitude'][i],traj_ped['longitude'][i]],
                       [traj_ped['latitude'][i+1],traj_ped['longitude'][i+1]]).km*60**2/delta_time
        speed_ped.append(speed)
for i in range(len(traj_veh)):
    if(i==len(traj_veh)-1):
        break
    if(traj_veh['id'][i]==traj_veh['id'][i+1]):
        speed=geodesic([traj_veh['latitude'][i],traj_veh['longitude'][i]],
                       [traj_veh['latitude'][i+1],traj_veh['longitude'][i+1]]).km*60**2/delta_time
        speed_veh.append(speed)
plt.figure(figsize=(12, 7), dpi=300)
plt.subplot(1,2,1)
plt.hist(speed_ped,bins=np.arange(0,25,0.1))
plt.title('Pedestrian Speed Distribution')
plt.ylabel('Counts')
plt.xlabel('Speed[km/h]')
plt.subplot(1,2,2)
plt.title('Vehicle Speed Distribution')
plt.ylabel('Counts')
plt.xlabel('Speed[km/h]')
plt.hist(speed_veh,bins=np.arange(0,70,0.1))
#plt.show()
fname='../images/stat.png'
plt.savefig(fname,dpi=300)
fname
