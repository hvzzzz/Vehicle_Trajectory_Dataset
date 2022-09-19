import numpy as np
import matplotlib.pyplot as plt
with open('bla.txt') as f:
    lines = f.readlines()
data=np.array(lines).astype(int)
with open('bla2.txt') as p:
    lines1 = p.readlines()
x=np.array(lines1).astype(int)
data=data/84
plt.plot(x,data,".");plt.xlabel("frame");plt.ylabel("numero de personas")
plt.grid(True);plt.savefig('pvst.png',dpi=300);plt.show()
