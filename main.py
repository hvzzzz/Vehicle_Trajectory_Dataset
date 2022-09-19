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
#plt.rcParams['figure.dpi']=200
#Homography Matrix
homograph=h_matrix('tools/Cal_PnP/data/calibration.txt')
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
def space_time_diagram():
    # Reading Trajectories from txt file
    header_list = ['frm','id','l','t','w','h','1','2','3','4','5'] 
    path="trajectories_txt/";names=os.listdir(path)
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
                outdir="data/trajectories/"+get_num_from_name(vid_name)+"/"+vid_name[:-4]+"/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir) 
                dfs.to_csv(outdir+"p"+j+".csv",index=False)  
"""
    # Region of Interest
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

    #Using homography matrix 
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
"""
def trayectory_processing():

    inv_homograph=np.linalg.inv(homograph)
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

    # Space Time Diagram Plots
    #fig1=plt.figure()
    #ax=plt.gca()
    #xfmt = md.DateFormatter('%H:%M:%S')
    #ax.xaxis.set_major_formatter(xfmt)
    #limit_time_d=dt.datetime(2022,2,5,6,30,00)
    #limit_time_u=dt.datetime(2022,2,5,6,32,0)
    #min_distance=4
    #for i in L_l3.keys():
    #  if(time_l3[i][0]>limit_time_d and time_l3[i][-1]<limit_time_u):
    #    if(np.max(L_l3[i][:,0])-np.min(L_l3[i][:,0])>min_distance):
    #      plt.plot(time_l3[i],L_l3[i][:,0],'^',markersize=0.1,label=i)
    #for i in L_l2.keys():
    #  if(time_l2[i][0]>limit_time_d and time_l2[i][-1]<limit_time_u):
    #    if(np.max(L_l2[i][:,0])-np.min(L_l2[i][:,0])>min_distance):
    #      plt.plot(time_l2[i],L_l2[i][:,0],'*',markersize=0.1,label=i)
    #for i in L_l1.keys():
    #  if(time_l1[i][0]>limit_time_d and time_l1[i][-1]<limit_time_u):
    #    if(np.max(L_l1[i][:,0])-np.min(L_l1[i][:,0])>min_distance):
    #      plt.plot(time_l1[i],L_l1[i][:,0],'.',markersize=0.1,label=i)
    ##plt.legend(loc='center right',prop={'size': 9},ncol=1)
    #plt.title('All Lanes')
    #plt.ylabel('Distance[m]')
    #plt.xlabel('Time')
    #plt.gcf().autofmt_xdate()
    #plt.grid(True)
    #plt.savefig('images/raw_spacetime.png',dpi=200)
    ##plt.show()

    #fig2=plt.figure(2)

    #epsi=0.2
    #L=geodesic((np.matmul(inv_homograph,np.append(p_9,1))/np.matmul(inv_homograph,np.append(p_9,1))[2])[:2],(np.matmul(inv_homograph,np.append(p_7,1))/np.matmul(inv_homograph,np.append(p_7,1))[2])[:2]).km*1e3-epsi
    ##print(L)
    #n_ob=10
    #dx=L/n_ob
    #ax=plt.gca()
    #xfmt = md.DateFormatter('%H:%M:%S')
    #ax.xaxis.set_major_formatter(xfmt)
    ##limit_time_d=dt.datetime(2022,2,5,6,0,00)
    ##limit_time_u=dt.datetime(2022,2,5,6,0,25)
    #limit_time_d=dt.datetime(2022,2,5,6,30,45)
    #limit_time_u=dt.datetime(2022,2,5,6,31,5)
    #min_distance=0
    #for i in L_l3.keys():
    #  if(time_l3[i][0]>limit_time_d and time_l3[i][-1]<limit_time_u):
    #    if(np.max(L_l3[i][:,0])-np.min(L_l3[i][:,0])>min_distance):
    #      plt.plot(time_l3[i],L_l3[i][:,0],'^',markersize=0.1,label=i)
    #      #plt.plot(time_l3[i],L_l3[i][:,0])
    ##for i in L_l2.keys():
    ##  if(time_l2[i][0]>limit_time_d and time_l2[i][-1]<limit_time_u):
    ##    if(np.max(L_l2[i][:,0])-np.min(L_l2[i][:,0])>min_distance):
    ##      plt.plot(time_l2[i],L_l2[i][:,0],'*',markersize=0.1,label=i)
    ##for i in L_l1.keys():
    ##  if(time_l1[i][0]>limit_time_d and time_l1[i][-1]<limit_time_u):
    ##    if(np.max(L_l1[i][:,0])-np.min(L_l1[i][:,0])>min_distance):
    ##      plt.plot(time_l1[i],L_l1[i][:,0],'o',markersize=0.5,label=i)
    #plt.legend(loc='lower right',prop={'size': 9},ncol=1)
    #plt.title('Lane 3')
    #plt.ylabel('Distance[m]')
    #plt.xlabel('Time')
    #plt.gcf().autofmt_xdate()
    #l=0
    #for i in range(n_ob):
    #  l=l+dx
    #  plt.axhline(l,linestyle='--',linewidth=0.5)
    #plt.savefig('images/raw_spacetime_zoom.png',dpi=200)
    ##plt.show()

    ## Speeds and Accelerations
    #speed_ac_l1=speeds(L_l1);time_speed_ac_l1=frm2time(speed_ac_l1);speed_ac_l1=speeds(L_l1)
    #speed_ac_l2=speeds(L_l2);time_speed_ac_l2=frm2time(speed_ac_l2);speed_ac_l2=speeds(L_l2)
    speed_ac_l3=speeds(L_l3);time_speed_ac_l3=frm2time(speed_ac_l3);speed_ac_l3=speeds(L_l3)

    #fig3=plt.figure(3) 

    #ax=plt.gca()
    #xfmt = md.DateFormatter('%H:%M:%S')
    #ax.xaxis.set_major_formatter(xfmt)
    #limit_time_d=dt.datetime(2022,2,5,6,30,45)
    #limit_time_u=dt.datetime(2022,2,5,6,31,5)
    #min_distance=0
    #for i in speed_ac_l3.keys():
    #  if(time_speed_ac_l3[i][0]>limit_time_d and time_speed_ac_l3[i][-1]<limit_time_u):
    #    #if(np.max(speed_ac_l3[i][:,0])-np.min(speed_ac_l3[i][:,0])>min_distance):
    #    plt.plot(time_speed_ac_l3[i],speed_ac_l3[i][:,1],label=i)
    #plt.legend()
    #plt.xlabel('Time')
    #plt.ylabel('$m/s^2$')
    ##plt.close(fig1);plt.close(fig2)
    #plt.savefig('images/accelerations.png',dpi=200)
    ##plt.show()

    # EMD 
    # Define signal

    t=speed_ac_l3['id_10752'][:,-1]
    s=speed_ac_l3['id_10752'][:,1]
    # Execute EMD on signal
    IMF = EMD().emd(s,t)
    N = IMF.shape[0]+1
    
    # Plot results
    #fig4=plt.figure(4)
    #plt.figure(figsize=(16, 20), dpi=150)
    #plt.subplot(N,1,1)
    #plt.plot(t, s, 'r')
    #plt.title("Input signal")
    #plt.xlabel("Frame")
    #plt.ylabel("Acceleration$[m/s^2]$")
    #for n, imf in enumerate(IMF):
    #    plt.subplot(N,1,n+2)
    #    plt.plot(t, imf, 'g')
    #    plt.title("IMF "+str(n+1))
    #    plt.xlabel("Frame")
    #plt.tight_layout()
    ##plt.savefig('images/acce_EMD.png')
    ##plt.close(fig1);plt.close(fig2);plt.close(fig3)
    #plt.show() 
    T=1/60
    N = IMF.shape[0]+1
    sos=signal.butter(10,0.3,'lowpass',fs=1/T,output='sos')
    filtered_signal=0
    for n, imf in enumerate(IMF):
      if(n<2):
        filtered_IMF=signal.sosfilt(sos,IMF[n])
        filtered_signal=filtered_signal+filtered_IMF
      else:
        filtered_signal=filtered_signal+IMF[n]
    # Speed
    h=1/60
    times=t
    df=filtered_signal
    I=np.zeros([len(df)+1])
    I[0]=speed_ac_l3['id_10752'][:,0][0]
    for k in range(len(df)):# integral
        if(k>0):
            I[k]=I[k-1]+h*df[k]
    I=np.roll(I,1)[1:]
    speed=I 
    # Position
    h=1/60
    times=t
    df=speed
    I=np.zeros([len(df)+1])
    I[0]=L_l3['id_10752'][:,0][0]
    for k in range(len(df)):# integral
      if(k>0):
        I[k]=I[k-1]+h*df[k]
    I=np.roll(I,1)[1:]
    #plt.figure(figsize=(16, 20), dpi=150)
    #plt.figure(figsize=(16, 20), dpi=150);plt.subplot(3,1,1)
    #plt.plot(speed_ac_l3['id_10752'][:,-1],speed_ac_l3['id_10752'][:,1]);plt.plot(t,filtered_signal)
    #plt.xlabel("Frame");plt.ylabel("$[m/s^2]$");plt.title('Acceleration');plt.subplot(3,1,2)
    #plt.plot(speed_ac_l3['id_10752'][:,-1],speed_ac_l3['id_10752'][:,0]);plt.plot(times,speed)
    #plt.xlabel("Frame");plt.ylabel("$[m/s]$");plt.title('Speed');plt.subplot(3,1,3)
    #plt.plot(L_l3['id_10752'][:,-1],L_l3['id_10752'][:,0]);plt.plot(times,I)
    #plt.xlabel("Frame");plt.ylabel("$[m]$");plt.title('Position')
    #plt.savefig('images/filtered_EMD.png')
    #plt.show()

    #Gaussian Kernel on Trayectory data
    x_sm=L_l3['id_10752'][:,-1]
    y_sm=L_l3['id_10752'][:,0]

    sigma = 8
    x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
    y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)
    fig, ax = plt.subplots(figsize=(10, 10))
    spl = interpolate.UnivariateSpline(x_sm, y_sm,k=5) 
    #plt.plot(x_sm, y_sm,'.',label='Original Data')
    #plt.plot(x_sm,y_g1d, 'magenta', linewidth=1,label='Gaussian Kernel Smoothing')
    #plt.plot(x_sm, spl(x_sm),'^',markersize=2,label='Univariante Spline')
    #plt.grid(True)
    #plt.title('Gaussian Kernel Applied to Trayectory')
    #plt.xlabel('Frame')
    #plt.ylabel('Distance$[m]$')
    ##plt.savefig('images/g_kernel_pos.png')
    #plt.legend()
    #plt.show()
    #Derivatives of filtered data
    dd_y=spds_acce(x_sm,spl(x_sm))  
    #i_p=np.zeros([1,2])
    i_p=[];i_temp=0
    for i,elem in enumerate(dd_y[:,1]):
        if(i>0 and i<len(dd_y[:,1])):
            if((dd_y[:,1][i-1]<0 and dd_y[:,1][i+1]>0) or (dd_y[:,1][i-1]<0 and dd_y[:,1][i+1]>0)):
                i_p.append([dd_y[:,2][i],elem])
                if(np.abs(i-i_temp)==1):
                    i_p.pop()
                i_temp=i
    i_px=np.where(dd_y[:,2]==i_p[0][0])
    #plt.subplot(1,2,1);plt.title('First Derivative')
    #plt.plot(dd_y[:,2],dd_y[:,0])
    #plt.plot(i_p[0][0],dd_y[:,0][i_px],'o')
    #plt.grid(True)
    #plt.subplot(1,2,2);plt.title('Second Derivative')
    #for i in i_p:
    #    plt.plot(i[0],i[1],'o') 
    #plt.plot(dd_y[:,2],dd_y[:,1],'o',markersize=2)
    #plt.grid(True)
    #plt.savefig('images/smoothed_data_derivatives.png') 
    #plt.show()
    #plt.plot(x_sm, spl(x_sm),linewidth=1,label='Univariante Spline')
    #plt.plot(i_p[0][0],spl(x_sm)[i_px],'o');plt.grid(True)
    #plt.savefig('images/inflection_point_trajectory.png') 
    #plt.show()
    temp_xm=x_sm[2:][dd_y[:,0]>0];temp_ym=spl(x_sm)[2:][dd_y[:,0]>0]
    boo=dd_y[:,0]>0
    indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1
    b = np.split(temp_ym, indices)
    b = b[0::2] if boo[0] else b[1::2]
    for i in range(len(b)):
        if(i>0):
            if(b[i][-1]<=b[i-1][-1]):
                b[i]=b[i][b[i]<=b[i-1][-1]]
    b = np.concatenate(b)
    index_xb=np.where(np.in1d(temp_ym,b))[0]
    x_b=np.zeros(len(temp_xm),dtype=bool)
    x_b[index_xb]=True
    x_b=temp_xm[x_b]
    y_spline =splines(x_sm,x_b,b) 
    #plt.subplot(1,3,1);plt.title('First Derivative',fontsize=7)
    #plt.plot(dd_y[:,2][dd_y[:,0]>0],dd_y[:,0][dd_y[:,0]>0],'.')
    #plt.grid(True);plt.subplot(1,3,2);plt.title('Trajectory without fallbacks',fontsize=7)
    #plt.plot(x_b,b,'.')
    #plt.grid(True)
    #plt.subplot(1,3,3);plt.title('Trajectory with fallbacks replaced by cubic spline',fontsize=7)
    #plt.plot(x_sm,y_spline,'.');plt.grid(True)
    ##plt.savefig('images/trajectory_with_no_fallbacks.png',dpi=200)  
    #plt.show()
    #plt.plot(x_b,b,'.')
    #plt.grid(True);plt.title('Trajectory without fallbacks',fontsize=7)
    #plt.show()
    #print(len(x_sm[2:][dd_y[:,0]>0]),len(spl(x_sm)[2:][dd_y[:,0]>0]),len(dd_y[:,0]>0))
    #print(len(x_sm[2:]),len(dd_y[:,0]))
    #print(type(b),type(temp_ylenen(b),len(temp_ym))
    #print(len(b),len(x_sm[b==temp_ym]))
    #print(dd_y[:,0]>0)
    sigma_p = 8
    y_g1d_p = ndimage.gaussian_filter1d(b, sigma_p)
    plt.plot(x_b,y_g1d_p,'.')
    plt.plot(x_b,b,'.')
    plt.show()
if __name__ == '__main__':
    space_time_diagram() 
    #trayectory_processing()
