import numpy as np
from shapely.geometry import Point

# DeepSort Trajectories Reading and Processing
def ltwh2midpoint(space):
    mid_pos=np.zeros([len(space),3])
    mid_pos[:,0]=space['id'].to_numpy('float')
    mid_pos[:,1]=space['l'].to_numpy('float')+space['w'].to_numpy(dtype='float')/2 #x_mid
    mid_pos[:,2]=space['t'].to_numpy('float')+space['h'].to_numpy(dtype='float')/2 #y_mid
    return mid_pos 

def ltwh2midpoint_t(space):
    mid_pos=np.zeros([len(space),4])
    mid_pos[:,0]=space['id'].to_numpy('float')
    mid_pos[:,1]=space['l'].to_numpy('float')+space['w'].to_numpy(dtype='float')/2 #x_mid
    mid_pos[:,2]=space['t'].to_numpy('float')+space['h'].to_numpy(dtype='float')/2 #y_mid
    mid_pos[:,3]=space['frm'].to_numpy('int')  
    return mid_pos

def order_dict(ways,point=False):
    num_ids=np.max(ways[:,0])
    ordered_tracks={}
    ordered_tracks_point_class={}
    for i in range(int(num_ids)):
      ordered_tracks['id_'+str(i+1)]=ways[ways[:,0]==(i+1),1:]
      if(point):
        for j in range(len(ordered_tracks['id_'+str(i+1)])):
          ordered_tracks_point_class['id_'+str(i+1)+'_'+str(j)]=Point(ordered_tracks['id_'+str(i+1)][j,:])
    if(not(point)):
      return int(num_ids),ordered_tracks
    else:
      return ordered_tracks_point_class

def n_order_dict(ways,point=False):  
    num_ids=np.unique(ways[:,0].astype(int))
    ordered_tracks={}
    ordered_tracks_point_class={}
    count=0
    for i in num_ids:
      count=count+1
      ordered_tracks['id_'+str(i)]=ways[ways[:,0]==(i),1:] 
    if(point):
      for j in range(len(ways)):
        ordered_tracks_point_class['id_'+str(j)]=Point(ways[j,1:])
      return ordered_tracks_point_class
    return num_ids,ordered_tracks

def tracks_in_area(p_i_c_p,mid_pos,polygon):
    bools=[]
    for i in p_i_c_p:
        bools.append(polygon.contains(p_i_c_p[str(i)]))
    mid_pos_in_area=mid_pos[np.array(bools),:]
    n ,ordered_tracks_in_area=n_order_dict(mid_pos_in_area)
    return n, ordered_tracks_in_area

# Vector Operations

def mod(v):
    modulo=np.sqrt(v[0]**2+v[1]**2)
    return modulo
  
def mod_c(v):
    modulo=np.sqrt(v[:,0]**2+v[:,1]**2)
    return modulo

def cos_ang(v1,v2):
    angle=(v1[0]*v2[0]+v1[1]*v2[1])/(mod(v1)*mod(v2))
    return angle

def cos_ang_c(v1,v2):
    angle=(v1[:,0]*v2[:,0]+v1[:,1]*v2[:,1])/(mod_c(v1)*mod_c(v2))
    return angle

def distance_2p(p1,p2):
    d=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return d

def initial_point(ids_list,o_t_i_a,vect_l,direction_x,direction_y): # determinacion de la distancia del punto inicial del vector a la proyeccion del punto inicial de la trayectoria en el vector de direccion. 
    initial_p=[direction_x[0,0],direction_y[0,0]]
    print(initial_p)
    initial_x={}
    for i in ids_list:
      l=distance_2p(o_t_i_a['id_'+str(i)][0,:2],initial_p)
      v=o_t_i_a['id_'+str(i)][0,:2]-initial_p
      ang=cos_ang(vect_l,v)
      initial_x['id_'+str(i)]=[l*ang,o_t_i_a['id_'+str(i)][0,2]]
      #print(i,v,ang)
    return initial_x

def xy2vect(ids_list,xy_dict):#determinacion del vector de vectores
    ordered_tracks_vect={}
    count=0
    count_length_1=0
    for i in ids_list:
      if(len(xy_dict['id_'+str(i)])>1):
        ordered_tracks_vect['id_'+str(i)]=-xy_dict['id_'+str(i)][:len(xy_dict['id_'+str(i)])-1]+xy_dict['id_'+str(i)][1:]
        ordered_tracks_vect['id_'+str(i)][:,2]=xy_dict['id_'+str(i)][1:,2]
      else:
        ids_list=np.delete(ids_list,count-count_length_1)
        count_length_1=count_length_1+1
      count=count+1
    return ids_list, ordered_tracks_vect

def cos_ang_vect(ids_list,o_t_v,vect_l):#determinacion de vector de l, la segunda columna son datos fantasma
    ordered_tracks_vect_cos={}
    L={}
    proyections=o_t_v.copy()
    for i in ids_list:
      #print(i)
      L['id_'+str(i)]=mod_c(o_t_v['id_'+str(i)])
      ordered_tracks_vect_cos['id_'+str(i)]=cos_ang_c(o_t_v['id_'+str(i)][:,:2],vect_l*np.ones([np.shape(o_t_v['id_'+str(i)][:,:2])[0],np.shape(o_t_v['id_'+str(i)][:,:2])[1]]))
      ordered_tracks_vect_cos['id_'+str(i)]=np.nan_to_num(ordered_tracks_vect_cos['id_'+str(i)])
    for i in ids_list:
      proyections['id_'+str(i)][:,0]=L['id_'+str(i)]*ordered_tracks_vect_cos['id_'+str(i)]
      proyections['id_'+str(i)][:,0]=np.nan_to_num(proyections['id_'+str(i)][:,0])
      proyections['id_'+str(i)][:,1]=ordered_tracks_vect_cos['id_'+str(i)]
    return proyections#ordered_tracks_vect_cos
    
# Time Space Diagram Generation

def space_t_diagram(list_init,init,list_vect,vect):#generacion del vector de l sumados
    space={}
    for i in list_vect: 
      space['id_'+str(i)]=np.zeros([np.shape(vect['id_'+str(i)])[0],2])
      for j in range(len(vect['id_'+str(i)])):
        if(j>0):
          space['id_'+str(i)][j][0]=vect['id_'+str(i)][j,0]+np.sum(vect['id_'+str(i)][:j,0],axis=0)
          space['id_'+str(i)][j][1]=vect['id_'+str(i)][j,2]
        else:
          space['id_'+str(i)][j][0]=vect['id_'+str(i)][j,0]
          space['id_'+str(i)][j][1]=vect['id_'+str(i)][j,2]
    return space
  
def diagram_corrected(init_p,proyect):#generacion del vector de l sumados con la distancia al punto inicial 
    diagram=init_p.copy()
    for i in proyect.keys():
      diagram[i]=np.zeros([np.shape(proyect[i])[0]+1,2]) 
      diagram[i][0,:]=init_p[i]
      diagram[i][1:,0]=proyect[i][:,0]+init_p[i][0]*np.ones(np.shape(proyect[i][:,0]))
      diagram[i][1:,1]=proyect[i][:,1]
    return diagram

def gen_xy_in_line(first_p,last_p,space_time):#conversion del vector de l sumados a coordenadas en la recta del vector direccion
    gamma=np.arctan((last_p[1]-first_p[1])/(last_p[0]-first_p[0]))
    if(gamma<0):
      gamma=-gamma 
    xy=space_time.copy()
    for i in space_time.keys():
      xy[i]=np.zeros([np.shape(space_time[i])[0],4])    
      if(not(isinstance(space_time[i],list))):
        xy[i][:,0]=first_p[0]*np.ones([np.shape(xy[i])[0]])+space_time[i][:,0]*np.cos(gamma)
        xy[i][:,1]=first_p[1]*np.ones([np.shape(xy[i])[0]])+space_time[i][:,0]*np.sin(gamma)
        xy[i][:,2]=np.ones([np.shape(space_time[i])[0]])
        xy[i][:,3]=space_time[i][:,1]
      else:
        xy[i][:,0]=first_p[0]*np.ones([np.shape(xy[i])[0]])+space_time[i][0]*np.cos(gamma)
        xy[i][:,1]=first_p[1]*np.ones([np.shape(xy[i])[0]])+space_time[i][0]*np.sin(gamma)
        xy[i][:,2]=np.ones([np.shape(space_time[i])[0]])
        xy[i][:,3]=space_time[i][1]
    return xy

def pixel2real_world(homograph,xy,first_p):
  xy_prim=xy.copy()
  inv_homograph=np.linalg.inv(homograph)
  first=first_p.copy()
  first.append(1)
  real_first=np.matmul(inv_homograph,first)
  real_first=real_first/real_first[2]
  real_first=real_first.T
  for i in xy.keys(): 
    gps=np.matmul(inv_homograph,xy[i][:,:3].T)
    gps=gps/gps[2]
    gps=gps.T
    xy_prim[i][:,:3]=gps
  return real_first,xy_prim
