from tools.homograph_matrix_read import h_matrix
from tools.utils import *
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pickle
from shapely.geometry.polygon import Polygon
from PyEMD import EMD
from scipy import signal
from scipy import ndimage
from scipy import interpolate
p_1=[1570.245283018868,339.9433962264151]
p_2=[760.8113207547169,358.811320754717]
p_3=[377.79245283018867,1073.9056603773583]
p_4=[1909.8679245283017,1070.132075471698]
p_5=[1906.0943396226414,511.64150943396226]
cam_1=[p_1,p_2,p_3,p_4,p_5]
p_6=[4.2075471698113205,198.43396226415092]
p_7=[4.2075471698113205,1068.245283018868]
p_8=[1907.9811320754716,1066.3584905660377]
p_9=[1917.4150943396226,424.84905660377353]
p_10=[1570.245283018868,253.1509433962264]
cam_2=[p_6,p_7,p_8,p_9,p_10]
p_11=[462.6981132075471,177.67924528301887]
p_12=[7.981132075471699,500.3207547169811]
p_13=[11.754716981132074,1070.132075471698]
p_14=[1909.8679245283017,1068.245283018868]
p_15=[1915.5283018867924,128.62264150943395]
p_16=[1511.754716981132,100.32075471698113]
cam_13=[p_10,p_11,p_12,p_13,p_14,p_15,p_16]
p_17=[1.75,136.5]
p_18=[1.75,715.25]
p_19=[1276.75,714.0]
p_20=[1278.0,152.75]
p_21=[1278.0,152.75]
p_22=[995.5,45.25]
cam_3=[p_17,p_18,p_19,p_20,p_21,p_22]
p_23=[340.5,290.25]
p_24=[329.25,715.25]
p_25=[1275.5,714.0]
p_26=[1274.25,284.0]
p_27=[974.25,154.0]
p_28=[700.5,121.5]
cam_5=[p_23,p_24,p_25,p_26,p_27,p_28]

def space_time_diagram():
    # Reading Trajectories from txt file
    header_list = ['frm','id','l','t','w','h','1','2','3','4','5']
    #path="trajectories_txt/"
    #path="Results/trajectories_pedestrian/"
    path="Results/trajectories_vehicle/"
    names=os.listdir(path)
    img=cv2.imread('./tools/Cal_PnP/pic/frm.jpg')[..., ::-1]
    for vid_name in names:
        print("Processing "+vid_name)
        space=pd.read_csv(path+vid_name,sep=" ",names=header_list)
        mid_pos_t=ltwh2midpoint_t(space)
        #plt.plot(mid_pos_t[:,1],mid_pos_t[:,2],'.',markersize=0.1)
        #plt.imshow(img)
        #plt.show()
        # Region of Interest
        ROI= Polygon(globals()["cam_"+get_num_from_name(vid_name)])

        # Trayectory in ROI
        points_in_class_point_ROI=n_order_dict(mid_pos_t,True)
        num_ids_in_area_ROI,ordered_tracks_in_area_ROI=tracks_in_area(points_in_class_point_ROI,mid_pos_t,ROI)
        #for i in num_ids_in_area_ROI:
        #    plt.plot(ordered_tracks_in_area_ROI[str(i)][:,0],ordered_tracks_in_area_ROI[str(i)][:,1],linewidth=1)
        #plt.imshow(img)
        #plt.savefig('images/trajectories_in_ROI.png',dpi=300)
        #plt.show()

        # Kalman Filter Export(export in format frame,id,x,y,type)
        for j in ordered_tracks_in_area_ROI.keys():
            if(len(ordered_tracks_in_area_ROI[j][:,2])>100):
                tp=[]
                for h in range(len(ordered_tracks_in_area_ROI[j][:,2])):
                    tp.append('ped')
                dfs=pd.DataFrame({"frame":ordered_tracks_in_area_ROI[j][:,2].astype(int),"id":int(j)*np.ones([len(ordered_tracks_in_area_ROI[j][:,2])]).astype(int),"x":ordered_tracks_in_area_ROI[j][:,0],"y":ordered_tracks_in_area_ROI[j][:,1],"type":tp})
                #outdir="data/trajectories/"+"p"+get_num_from_name(vid_name)+"/"+vid_name[:-4]+"/"
                outdir="data/trajectories/"+"v"+get_num_from_name(vid_name)+"/"+vid_name[:-4]+"/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                dfs.to_csv(outdir+"p"+j+".csv",index=False)
if __name__ == '__main__':
    space_time_diagram()
