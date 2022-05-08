from tools.homograph_matrix_read import h_matrix
from tools.utils import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import cv2
import pickle
from shapely.geometry.polygon import Polygon
plt.rcParams['figure.dpi']=200

def space_time_diagram(): 
    # Reading Trajectories from txt file
    header_list = ['frm','id','l','t','w','h','1','2','3','4','5'] 
    space=pd.read_csv('deepsort/Universitaria-LaCultura-carrildebajada_2022-02-05055848-2022-02-06000003_.txt',sep=" ",names=header_list) 
    mid_pos_t=ltwh2midpoint_t(space)    

    # Region of Insterest
    p_1=[713.0731707317073,461.92682926829264]
    p_2=[1155.7560975609756,446.07317073170725]
    p_3=[995.9999999999999,93.6341463414634]          
    p_4=[943.560975609756,99.73170731707316]  
    p_5=[841.1219512195121,459.4878048780488]
    p_6=[959.4146341463414,97.29268292682926]
    p_7=[1005.7560975609755,448.5121951219512]
    p_8=[980.1463414634146,93.6341463414634]
    lane_1= Polygon([p_1,p_5,p_6,p_4])
    lane_2= Polygon([p_5,p_7,p_8,p_6])
    lane_3= Polygon([p_7,p_2,p_3,p_8])

    # Trayectories in lanes
    points_in_class_point_l1=n_order_dict(mid_pos_t,True)
    num_ids_in_area_l1,ordered_tracks_in_area_l1=tracks_in_area(points_in_class_point_l1,mid_pos_t,lane_1)
    points_in_class_point_l2=n_order_dict(mid_pos_t,True)
    num_ids_in_area_l2,ordered_tracks_in_area_l2=tracks_in_area(points_in_class_point_l2,mid_pos_t,lane_2)
    points_in_class_point_l3=n_order_dict(mid_pos_t,True)
    num_ids_in_area_l3,ordered_tracks_in_area_l3=tracks_in_area(points_in_class_point_l3,mid_pos_t,lane_3)

    #Vector Extraction

    p_9=[976.4878048780487,66.80487804878047]  
    p_10=[942.3414634146341,1058.2682926829268]
    direction_x=np.zeros([2,1])
    direction_x[0,0]=p_9[0];direction_x[1,0]=p_10[0]  
    direction_y=np.zeros([2,1])
    direction_y[0,0]=p_9[1];direction_y[1,0]=p_10[1]
    v_l1=np.copy(p_10)-np.copy(p_9)

    #Lane 1
    initial_x_l1=initial_point(num_ids_in_area_l1,ordered_tracks_in_area_l1,v_l1,direction_x,direction_y)
    ids_vect_l1, ordered_tracks_vect_l1=xy2vect(num_ids_in_area_l1,ordered_tracks_in_area_l1) 
    proyections_l1=cos_ang_vect(ids_vect_l1, ordered_tracks_vect_l1,v_l1)
    diagram_l1=space_t_diagram(num_ids_in_area_l1,initial_x_l1,ids_vect_l1,proyections_l1)
    space_time_l1=diagram_corrected(initial_x_l1,diagram_l1)

    # Lane 2
    initial_x_l2=initial_point(num_ids_in_area_l2,ordered_tracks_in_area_l2,v_l1,direction_x,direction_y)
    ids_vect_l2, ordered_tracks_vect_l2=xy2vect(num_ids_in_area_l2,ordered_tracks_in_area_l2)
    proyections_l2=cos_ang_vect(ids_vect_l2, ordered_tracks_vect_l2,v_l1)
    diagram_l2=space_t_diagram(num_ids_in_area_l2,initial_x_l2,ids_vect_l2,proyections_l2)
    space_time_l2=diagram_corrected(initial_x_l2,diagram_l2)

    #Lane 3
    initial_x_l3=initial_point(num_ids_in_area_l3,ordered_tracks_in_area_l3,v_l1,direction_x,direction_y)
    ids_vect_l3, ordered_tracks_vect_l3=xy2vect(num_ids_in_area_l3,ordered_tracks_in_area_l3)
    proyections_l3=cos_ang_vect(ids_vect_l3, ordered_tracks_vect_l3,v_l1)
    diagram_l3=space_t_diagram(num_ids_in_area_l3,initial_x_l3,ids_vect_l3,proyections_l3)
    space_time_l3=diagram_corrected(initial_x_l3,diagram_l3)

    #Homography Matrix

    homograph=h_matrix('tools/Cal_PnP/data/calibration.txt')

    #Lane 1
    xy_l1=gen_xy_in_line(p_9,p_10,space_time_l1)    
    vect_first_l1,xy_prim_l1=pixel2real_world(homograph,xy_l1,p_9)
    L_l1=L_in_realworld(vect_first_l1,xy_prim_l1,'great_circle')

    #Lane 2
    xy_l2=gen_xy_in_line(p_9,p_10,space_time_l2)
    vect_first_l2,xy_prim_l2=pixel2real_world(homograph,xy_l2,p_9)
    L_l2=L_in_realworld(vect_first_l2,xy_prim_l2,'great_circle')

    #Lane 3
    xy_l3=gen_xy_in_line(p_9,p_10,space_time_l3)
    vect_first_l3,xy_prim_l3=pixel2real_world(homograph,xy_l3,p_9)
    L_l3=L_in_realworld(vect_first_l3,xy_prim_l3,'great_circle')

    with open('files/L_l1.pkl', 'wb') as f:
        pickle.dump(L_l1, f)
    with open('files/L_l2.pkl', 'wb') as f:
        pickle.dump(L_l2, f)
    with open('files/L_l3.pkl', 'wb') as f:
        pickle.dump(L_l3, f)

def trayectory_processing():

    with open('files/L_l1.pkl', 'rb') as f:
        L_l1 = pickle.load(f);time_l1=frm2time(L_l1)
    with open('files/L_l1.pkl', 'rb') as f:
        L_l1 = pickle.load(f)
    with open('files/L_l2.pkl', 'rb') as f:
        L_l2 = pickle.load(f);time_l2=frm2time(L_l2)
    with open('files/L_l2.pkl', 'rb') as f:
        L_l2 = pickle.load(f)
    with open('files/L_l3.pkl', 'rb') as f:
        L_l3 = pickle.load(f);time_l3=frm2time(L_l3)
    with open('files/L_l3.pkl', 'rb') as f:
        L_l3 = pickle.load(f) 
    ax=plt.gca()
    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    limit_time_d=dt.datetime(2022,2,5,6,30,00)
    limit_time_u=dt.datetime(2022,2,5,6,32,0)
    min_distance=4
    for i in L_l3.keys():
      if(time_l3[i][0]>limit_time_d and time_l3[i][-1]<limit_time_u):
        if(np.max(L_l3[i][:,0])-np.min(L_l3[i][:,0])>min_distance):
          plt.plot(time_l3[i],L_l3[i][:,0],'^',markersize=0.1,label=i)
    for i in L_l2.keys():
      if(time_l2[i][0]>limit_time_d and time_l2[i][-1]<limit_time_u):
        if(np.max(L_l2[i][:,0])-np.min(L_l2[i][:,0])>min_distance):
          plt.plot(time_l2[i],L_l2[i][:,0],'*',markersize=0.1,label=i)
    for i in L_l1.keys():
      if(time_l1[i][0]>limit_time_d and time_l1[i][-1]<limit_time_u):
        if(np.max(L_l1[i][:,0])-np.min(L_l1[i][:,0])>min_distance):
          plt.plot(time_l1[i],L_l1[i][:,0],'.',markersize=0.1,label=i)
    #plt.legend(loc='center right',prop={'size': 9},ncol=1)
    plt.title('All Lanes')
    plt.ylabel('Distance[m]')
    plt.xlabel('Time')
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.savefig('images/raw_spacetime.png',dpi=200)
    plt.show()
if __name__ == '__main__':
    #space_time_diagram() 
    trayectory_processing()
