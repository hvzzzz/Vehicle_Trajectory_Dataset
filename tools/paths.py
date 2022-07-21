import os
path="../trajectories_txt/"
names=os.listdir(path)
for i in names:
    print(path+i[:-4])
