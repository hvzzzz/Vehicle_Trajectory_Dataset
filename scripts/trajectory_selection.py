import geopandas as gpd
from geopandas import GeoDataFrame
import sys
sys.path.append("/home/laad/Projects/hanan/Vehicle_Trajectory_Dataset/")
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

ids_p=[371,228,764,18,31,266,499,496,297,516,654,290,572,124,536,655,576,609,290,310,203,577,474,542,641,551,474,705,619,603,123,2,1,162,11,26,30,174,3,17,472,149,2,329,519,395,152,216,45,325,3,4,241,280,207,345,167,183,458,284,399,408,332,231,185,73,52,158,72,146,129,256,80,206,1,173,186,263,270,117,134,220,81,61,308,209,144,222,108,146,100,507,595,357,6,430,294,496,620,531,500,551,497,521,325,490,613,504,650,272,765,713,742,363,300,363,448,117,653,154,773,584,371,77,592,590,655,706,611,624,775,654,272,344,652,699,456,481,368,692,446,510,553,264,400,577,567,430,568,448,459,679,469,508,499,659,769,145,565,716,519,564,531,391,194,408,614,464,290,425,443]
ids_v=[150,85,99,9,137,122,140,161,136,158,190,182,167,1,202,214,167,195,206,199,143,171,177,104,178,186,185,212,202,168,39,98,106,139,1,79,32,104,29,26,194,11,139,178,92,171,132,147,37,178,95,24,123,142,158,213,205,182,250,269,241,56,42,198,127,105,54,59,45,120,92,169,78,102,18,157,89,150,151,72,192,129,131,100,203,120,115,125,98,118,94,130,131,123,110,134,140,161,172,139,143,227,212,227,180,210,177,159,190,92,184,184,184,175,168,176,180,1,160,154,156,138,181,129,209,209,214,215,218,241,265,241,96,135,167,174,185,185,183,201,205,221,228,132,198,171,183,177,194,143,147,221,185,196,161,201,215,88,212,227,208,218,168,138,129,273,136,192,116,107,94]

interaction_p={}
#interaction_p['1_06 13 00']=ids_p[0]
interaction_p['1_06 37 00']=ids_p[1]
interaction_p['1_07 01 00']=ids_p[2]
interaction_p['5_04:37:00']=ids_p[3]
interaction_p['5_04:57:00']=ids_p[4]
interaction_p['13_03:37:00']=ids_p[5]
interaction_p['3_04:25:00']=ids_p[6]
interaction_p['3_04:37:00']=ids_p[7]
interaction_p['3_04:41:00']=ids_p[8]
interaction_p['3_04:45:00']=ids_p[9]
interaction_p['3_05:05:00']=ids_p[10]
interaction_p['3_05:21:00']=ids_p[11]
interaction_p['3_04:45:01']=ids_p[12]
interaction_p['3_05:33:00']=ids_p[13]
interaction_p['3_05:53:00']=ids_p[14]
interaction_p['3_05:53:01']=ids_p[15]
interaction_p['3_06:01:00']=ids_p[16]
interaction_p['3_06:09:00']=ids_p[17]
interaction_p['3_06:25:00']=ids_p[18]
interaction_p['3_06:33:00']=ids_p[19]
interaction_p['3_06:45:00']=ids_p[20]
interaction_p['3_06:49:00']=ids_p[21]
interaction_p['3_06:53:00']=ids_p[22]
interaction_p['3_06:57:00']=ids_p[23]
interaction_p['3_07:01:00']=ids_p[24]
interaction_p['3_07:05:00']=ids_p[25]
interaction_p['3_07:09:00']=ids_p[26]
interaction_p['3_07:25:00']=ids_p[27]
interaction_p['3_07:41:00']=ids_p[28]
interaction_p['3_07:45:00']=ids_p[29]
interaction_p['3_08:29:00']=ids_p[30]
interaction_p['5_05:45:00']=ids_p[31]
interaction_p['5_05:57:00']=ids_p[32]
interaction_p['13_03:41:00']=ids_p[33]
interaction_p['13_03:49:00']=ids_p[34]
interaction_p['13_03:53:00']=ids_p[35]
interaction_p['13_03:57:00']=ids_p[36]
interaction_p['13_04:05:00']=ids_p[37]
interaction_p['13_04:09:00']=ids_p[38]
interaction_p['13_04:13:00']=ids_p[39]
#interaction_p['1_06:01:00']=ids_p[40]
interaction_p['1_07:29:00']=ids_p[41]
#interaction_p['1_04:45:00']=ids_p[42]
#interaction_p['1_04:53:00']=ids_p[43]
interaction_p['1_04:57:00']=ids_p[44]
interaction_p['1_05:05:00']=ids_p[45]
#interaction_p['1_05:13:00']=ids_p[46]
#interaction_p['1_05:37:00']=ids_p[47]
#interaction_p['1_05:37:01']=ids_p[48]
#interaction_p['1_05:53:00']=ids_p[49]
interaction_p['1_05:57:00']=ids_p[50]
interaction_p['1_04:17:00']=ids_p[51]
#interaction_p['1_04:25:00']=ids_p[52]
#interaction_p['1_04:41:00']=ids_p[53]#
interaction_p['2_06:13:00']=ids_p[54]
interaction_p['2_06:41:00']=ids_p[55]
interaction_p['2_06:45:00']=ids_p[56]
interaction_p['2_06:57:00']=ids_p[57]
interaction_p['2_07:01:00']=ids_p[58]
interaction_p['2_07:09:00']=ids_p[59]
interaction_p['2_07:21:00']=ids_p[60]
interaction_p['2_07:53:00']=ids_p[61]
interaction_p['2_10:57:00']=ids_p[62]
interaction_p['2_12:01:00']=ids_p[63]
interaction_p['7_04:57:00']=ids_p[64]
interaction_p['7_05:17:00']=ids_p[65]
interaction_p['7_05:45:00']=ids_p[66]
interaction_p['7_05:49:00']=ids_p[67]
interaction_p['7_05:57:00']=ids_p[68]
interaction_p['7_06:41:00']=ids_p[69]
interaction_p['7_06:49:00']=ids_p[70]
interaction_p['7_06:53:00']=ids_p[71]
interaction_p['7_06:57:00']=ids_p[72]
interaction_p['7_07:21:00']=ids_p[73]
interaction_p['7_07:41:00']=ids_p[74]
#interaction_p['7_07:49:00']=ids_p[75]
interaction_p['7_08:01:00']=ids_p[76]
interaction_p['7_08:05:00']=ids_p[77]
interaction_p['7_08:09:00']=ids_p[78]
interaction_p['7_08:25:00']=ids_p[79]
interaction_p['7_08:37:00']=ids_p[80]
interaction_p['7_08:41:00']=ids_p[81]
interaction_p['7_08:45:00']=ids_p[82]
interaction_p['7_09:09:00']=ids_p[83]
interaction_p['9_08:09:00']=ids_p[84]
interaction_p['9_08:13:00']=ids_p[85]
interaction_p['9_08:17:00']=ids_p[86]
interaction_p['9_08:21:00']=ids_p[87]
interaction_p['9_08:45:00']=ids_p[88]
interaction_p['9_08:53:00']=ids_p[89]
interaction_p['9_09:05:00']=ids_p[90]
interaction_p['3_04:17:00']=ids_p[91]
interaction_p['3_04:17:01']=ids_p[92]
interaction_p['3_04:17:02']=ids_p[93]
interaction_p['3_04:25:01']=ids_p[94]
interaction_p['3_04:25:02']=ids_p[95]
interaction_p['3_04:37:01']=ids_p[96]
interaction_p['3_04:37:02']=ids_p[97]
interaction_p['3_04:45:02']=ids_p[98]
interaction_p['3_04:49:00']=ids_p[99]
interaction_p['3_04:49:01']=ids_p[100]
interaction_p['3_04:53:00']=ids_p[101]
interaction_p['3_04:57:00']=ids_p[102]
interaction_p['3_04:57:01']=ids_p[103]
interaction_p['3_05:01:00']=ids_p[104]
interaction_p['3_05:01:01']=ids_p[105]
interaction_p['3_05:05:01']=ids_p[106]
interaction_p['3_05:05:02']=ids_p[107]
interaction_p['3_05:05:03']=ids_p[108]
interaction_p['3_05:17:00']=ids_p[109]
interaction_p['3_05:17:01']=ids_p[110]
interaction_p['3_05:17:02']=ids_p[111]
interaction_p['3_05:17:03']=ids_p[112]
interaction_p['3_05:29:00']=ids_p[113]
interaction_p['3_05:29:01']=ids_p[114]
interaction_p['3_05:29:02']=ids_p[115]
interaction_p['3_05:29:03']=ids_p[116]
interaction_p['3_05:33:01']=ids_p[117]
interaction_p['3_05:37:00']=ids_p[118]
#interaction_p['3_05:41:00']=ids_p[119]
interaction_p['3_05:41:01']=ids_p[120]
interaction_p['3_05:41:02']=ids_p[121]
interaction_p['3_05:53:02']=ids_p[122]
interaction_p['3_05:53:03']=ids_p[123]
interaction_p['3_05:53:04']=ids_p[124]
interaction_p['3_05:53:05']=ids_p[125]
interaction_p['3_05:53:06']=ids_p[126]
interaction_p['3_05:53:07']=ids_p[127]
interaction_p['3_05:53:08']=ids_p[128]
interaction_p['3_05:57:00']=ids_p[129]
interaction_p['3_05:57:01']=ids_p[130]
interaction_p['3_05:57:02']=ids_p[131]
interaction_p['3_06:01:01']=ids_p[132]
interaction_p['3_06:01:02']=ids_p[133]
interaction_p['3_06:01:03']=ids_p[134]
interaction_p['3_06:01:04']=ids_p[135]
interaction_p['3_06:05:00']=ids_p[136]
interaction_p['3_06:05:01']=ids_p[137]
interaction_p['3_06:09:01']=ids_p[138]
interaction_p['3_06:09:02']=ids_p[139]
interaction_p['3_06:17:00']=ids_p[140]
interaction_p['3_06:17:01']=ids_p[141]
interaction_p['3_06:25:02']=ids_p[142]
interaction_p['3_06:41:00']=ids_p[143]
interaction_p['3_06:41:01']=ids_p[144]
interaction_p['3_06:49:01']=ids_p[145]
interaction_p['3_06:49:02']=ids_p[146]
interaction_p['3_06:53:01']=ids_p[147]
interaction_p['3_06:53:02']=ids_p[148]
interaction_p['3_07:01:01']=ids_p[149]
interaction_p['3_07:01:02']=ids_p[150]
interaction_p['3_07:05:01']=ids_p[151]
interaction_p['3_07:09:01']=ids_p[152]
interaction_p['3_07:09:02']=ids_p[153]
interaction_p['3_07:13:00']=ids_p[154]
interaction_p['3_07:21:00']=ids_p[155]
interaction_p['3_07:21:01']=ids_p[156]
interaction_p['3_07:25:01']=ids_p[157]
interaction_p['3_07:25:02']=ids_p[158]
interaction_p['3_07:25:03']=ids_p[159]
interaction_p['3_07:41:01']=ids_p[160]
interaction_p['3_07:41:02']=ids_p[161]
interaction_p['3_07:45:01']=ids_p[162]
interaction_p['3_07:45:02']=ids_p[163]
interaction_p['3_07:53:00']=ids_p[164]
interaction_p['3_08:05:00']=ids_p[165]
interaction_p['3_08:13:00']=ids_p[166]
interaction_p['3_08:17:00']=ids_p[167]
interaction_p['3_08:25:00']=ids_p[168]
interaction_p['3_08:25:01']=ids_p[169]
interaction_p['3_08:29:01']=ids_p[170]


interaction_v={}
#interaction_v['1_06_13_00']=ids_v[0]
interaction_v['1_06_37_00']=ids_v[1]
interaction_v['1_07_01_00']=ids_v[2]
interaction_v['5_04:37:00']=ids_v[3]
interaction_v['5_04:57:00']=ids_v[4]
interaction_v['13_03:37:00']=ids_v[5]
interaction_v['3_04:25:00']=ids_v[6]
interaction_v['3_04:37:00']=ids_v[7]
interaction_v['3_04:41:00']=ids_v[8]
interaction_v['3_04:45:00']=ids_v[9]
interaction_v['3_05:05:00']=ids_v[10]
interaction_v['3_05:21:00']=ids_v[11]
interaction_v['3_04:45:01']=ids_v[12]
interaction_v['3_05:33:00']=ids_v[13]
interaction_v['3_05:53:00']=ids_v[14]
interaction_v['3_05:53:01']=ids_v[15]
interaction_v['3_06:01:00']=ids_v[16]
interaction_v['3_06:09:00']=ids_v[17]
interaction_v['3_06:25:00']=ids_v[18]
interaction_v['3_06:33:00']=ids_v[19]
interaction_v['3_06:45:00']=ids_v[20]
interaction_v['3_06:49:00']=ids_v[21]
interaction_v['3_06:53:00']=ids_v[22]
interaction_v['3_06:57:00']=ids_v[23]
interaction_v['3_07:01:00']=ids_v[24]
interaction_v['3_07:05:00']=ids_v[25]
interaction_v['3_07:09:00']=ids_v[26]
interaction_v['3_07:25:00']=ids_v[27]
interaction_v['3_07:41:00']=ids_v[28]
interaction_v['3_07:45:00']=ids_v[29]
interaction_v['3_08:29:00']=ids_v[30]
interaction_v['5_05:45:00']=ids_v[31]
interaction_v['5_05:57:00']=ids_v[32]
interaction_v['13_03:41:00']=ids_v[33]
interaction_v['13_03:49:00']=ids_v[34]
interaction_v['13_03:53:00']=ids_v[35]
interaction_v['13_03:57:00']=ids_v[36]
interaction_v['13_04:05:00']=ids_v[37]
interaction_v['13_04:09:00']=ids_v[38]
interaction_v['13_04:13:00']=ids_v[39]
#interaction_v['1_06:01:00']=ids_v[40]
interaction_v['1_07:29:00']=ids_v[41]
#interaction_v['1_04:45:00']=ids_v[42]
#interaction_v['1_04:53:00']=ids_v[43]
interaction_v['1_04:57:00']=ids_v[44]
interaction_v['1_05:05:00']=ids_v[45]
#interaction_v['1_05:13:00']=ids_v[46]
#interaction_v['1_05:37:00']=ids_v[47]
#interaction_v['1_05:37:01']=ids_v[48]
#interaction_v['1_05:53:00']=ids_v[49]
interaction_v['1_05:57:00']=ids_v[50]
interaction_v['1_04:17:00']=ids_v[51]
#interaction_v['1_04:25:00']=ids_v[52]
#interaction_v['1_04:41:00']=ids_v[53]
interaction_v['2_06:13:00']=ids_v[54]
interaction_v['2_06:41:00']=ids_v[55]
interaction_v['2_06:45:00']=ids_v[56]
interaction_v['2_06:57:00']=ids_v[57]
interaction_v['2_07:01:00']=ids_v[58]
interaction_v['2_07:09:00']=ids_v[59]
interaction_v['2_07:21:00']=ids_v[60]
interaction_v['2_07:53:00']=ids_v[61]
interaction_v['2_10:57:00']=ids_v[62]
interaction_v['2_12:01:00']=ids_v[63]
interaction_v['7_04:57:00']=ids_v[64]
interaction_v['7_05:17:00']=ids_v[65]
interaction_v['7_05:45:00']=ids_v[66]
interaction_v['7_05:49:00']=ids_v[67]
interaction_v['7_05:57:00']=ids_v[68]
interaction_v['7_06:41:00']=ids_v[69]
interaction_v['7_06:49:00']=ids_v[70]
interaction_v['7_06:53:00']=ids_v[71]
interaction_v['7_06:57:00']=ids_v[72]
interaction_v['7_07:21:00']=ids_v[73]
interaction_v['7_07:41:00']=ids_v[74]
#interaction_v['7_07:49:00']=ids_v[75]
interaction_v['7_08:01:00']=ids_v[76]
interaction_v['7_08:05:00']=ids_v[77]
interaction_v['7_08:09:00']=ids_v[78]
interaction_v['7_08:25:00']=ids_v[79]
interaction_v['7_08:37:00']=ids_v[80]
interaction_v['7_08:41:00']=ids_v[81]
interaction_v['7_08:45:00']=ids_v[82]
interaction_v['7_09:09:00']=ids_v[83]
interaction_v['9_08:09:00']=ids_v[84]
interaction_v['9_08:13:00']=ids_v[85]
interaction_v['9_08:17:00']=ids_v[86]
interaction_v['9_08:21:00']=ids_v[87]
interaction_v['9_08:45:00']=ids_v[88]
interaction_v['9_08:53:00']=ids_v[89]
interaction_v['9_09:05:00']=ids_v[90]
interaction_v['3_04:17:00']=ids_v[91]
interaction_v['3_04:17:01']=ids_v[92]
interaction_v['3_04:17:02']=ids_v[93]
interaction_v['3_04:25:01']=ids_v[94]
interaction_v['3_04:25:02']=ids_v[95]
interaction_v['3_04:37:01']=ids_v[96]
interaction_v['3_04:37:02']=ids_v[97]
interaction_v['3_04:45:02']=ids_v[98]
interaction_v['3_04:49:00']=ids_v[99]
interaction_v['3_04:49:01']=ids_v[100]
interaction_v['3_04:53:00']=ids_v[101]
interaction_v['3_04:57:00']=ids_v[102]
interaction_v['3_04:57:01']=ids_v[103]
interaction_v['3_05:01:00']=ids_v[104]
interaction_v['3_05:01:01']=ids_v[105]
interaction_v['3_05:05:01']=ids_v[106]
interaction_v['3_05:05:02']=ids_v[107]
interaction_v['3_05:05:03']=ids_v[108]
interaction_v['3_05:17:00']=ids_v[109]
interaction_v['3_05:17:01']=ids_v[110]
interaction_v['3_05:17:02']=ids_v[111]
interaction_v['3_05:17:03']=ids_v[112]
interaction_v['3_05:29:00']=ids_v[113]
interaction_v['3_05:29:01']=ids_v[114]
interaction_v['3_05:29:02']=ids_v[115]
interaction_v['3_05:29:03']=ids_v[116]
interaction_v['3_05:33:01']=ids_v[117]
interaction_v['3_05:37:00']=ids_v[118]
#interaction_v['3_05:41:00']=ids_v[119]
interaction_v['3_05:41:01']=ids_v[120]
interaction_v['3_05:41:02']=ids_v[121]
interaction_v['3_05:53:02']=ids_v[122]
interaction_v['3_05:53:03']=ids_v[123]
interaction_v['3_05:53:04']=ids_v[124]
interaction_v['3_05:53:05']=ids_v[125]
interaction_v['3_05:53:06']=ids_v[126]
interaction_v['3_05:53:07']=ids_v[127]
interaction_v['3_05:53:08']=ids_v[128]
interaction_v['3_05:57:00']=ids_v[129]
interaction_v['3_05:57:01']=ids_v[130]
interaction_v['3_05:57:02']=ids_v[131]
interaction_v['3_06:01:01']=ids_v[132]
interaction_v['3_06:01:02']=ids_v[133]
interaction_v['3_06:01:03']=ids_v[134]
interaction_v['3_06:01:04']=ids_v[135]
interaction_v['3_06:05:00']=ids_v[136]
interaction_v['3_06:05:01']=ids_v[137]
interaction_v['3_06:09:01']=ids_v[138]
interaction_v['3_06:09:02']=ids_v[139]
interaction_v['3_06:17:00']=ids_v[140]
interaction_v['3_06:17:01']=ids_v[141]
interaction_v['3_06:25:02']=ids_v[142]
interaction_v['3_06:41:00']=ids_v[143]
interaction_v['3_06:41:01']=ids_v[144]
interaction_v['3_06:49:01']=ids_v[145]
interaction_v['3_06:49:02']=ids_v[146]
interaction_v['3_06:53:01']=ids_v[147]
interaction_v['3_06:53:02']=ids_v[148]
interaction_v['3_07:01:01']=ids_v[149]
interaction_v['3_07:01:02']=ids_v[150]
interaction_v['3_07:05:01']=ids_v[151]
interaction_v['3_07:09:01']=ids_v[152]
interaction_v['3_07:09:02']=ids_v[153]
interaction_v['3_07:13:00']=ids_v[154]
interaction_v['3_07:21:00']=ids_v[155]
interaction_v['3_07:21:01']=ids_v[156]
interaction_v['3_07:25:01']=ids_v[157]
interaction_v['3_07:25:02']=ids_v[158]
interaction_v['3_07:25:03']=ids_v[159]
interaction_v['3_07:41:01']=ids_v[160]
interaction_v['3_07:41:02']=ids_v[161]
interaction_v['3_07:45:01']=ids_v[162]
interaction_v['3_07:45:02']=ids_v[163]
interaction_v['3_07:53:00']=ids_v[164]
interaction_v['3_08:05:00']=ids_v[165]
interaction_v['3_08:13:00']=ids_v[166]
interaction_v['3_08:17:00']=ids_v[167]
interaction_v['3_08:25:00']=ids_v[168]
interaction_v['3_08:25:01']=ids_v[169]
interaction_v['3_08:29:01']=ids_v[170]

cam_list=[1,3,5,13,2,7,9]
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
homograph_2=h_matrix('../tools/Cal_PnP/data/cam_2/calibration.txt')
homograph_7=h_matrix('../tools/Cal_PnP/data/cam_7/calibration.txt')
homograph_9=h_matrix('../tools/Cal_PnP/data/cam_9/calibration.txt')

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
            print("Processing Pedestrian "+vid_name)
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
#print(len(interaction_v),len(interaction_p))
#print(interaction_v['1_04:17:00'],interaction_p['1_04:17:00'])
for (k,v), (k2,v2) in zip(interaction_v.items(), interaction_p.items()):
    print(k)
    #print(len(v),len(v2))
    plt.plot(v[:,0],v[:,1],label='Vehicle')
    plt.plot(v2[:,0],v2[:,1],label='Pedestrian')
    plt.title(k+" "+k2)
    plt.legend()
    fname='../images/'+k+'_'+k2+'.png'
    plt.imshow(img)
    #plt.savefig(fname)
    #plt.show()

#print(get_num_from_name())
#print(globals()["homograph_"+str(1)])
#print(list(interaction_p_temp.keys())[6])
#print(get_num_from_name(list(interaction_p_temp.keys())[6]))

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
    gps=np.matmul(np.linalg.inv(globals()["homograph_"+str(get_num_from_name(i))]),interaction_p_temp[i][:,:3].T)
    gps=gps/gps[2]
    gps=gps.T
    interaction_p_gps[i]=gps[:,:2]

for i in interaction_v_temp.keys():
    #print(np.shape(interaction_v[i]))
    interaction_v_temp[i]=np.append(interaction_v_temp[i],np.ones([len(interaction_v_temp[i]),1]),axis=1)
    #print(np.shape(inv_homograph),np.shape(interaction_v[i]))
    gps=np.matmul(np.linalg.inv(globals()["homograph_"+str(get_num_from_name(i))]),interaction_v_temp[i][:,:3].T)
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
