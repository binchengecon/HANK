import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



Data_Dir = "./Python/data/"
# os.makedirs(Data_Dir, exist_ok=True)

with open(Data_Dir+ "model_result", "rb") as f:
   res = pickle.load(f)

F_init = res["F_init"]
a_star = res["a_star"]
c_star = res["c_star"]

# error = np.sum(F_init[:,0,0]-F_init[:,1,1],axis=0)

F = F_init[:,0,0]
a = a_star[:,0,0]
c = c_star[:,0,0]







W1_min = 0.0
W1_max = 1.0
hW1 = 0.01
W1 = np.arange(W1_min,W1_max+hW1,hW1)
nW1 = len(W1)

W2_min = 0.0
W2_max = 1.0
hW2 = 0.04
W2 = np.arange(W2_min,W2_max+hW2,hW2)
nW2 = len(W2)

W3_min = 0.0
W3_max = 1.0
hW3 = 0.04
W3 = np.arange(W3_min,W3_max+hW3,hW3)
nW3 = len(W3)


F_init = res["F_init"]
a_star = res["a_star"]
c_star = res["c_star"]

# error = np.sum(F_init[:,0,0]-F_init[:,1,1],axis=0)

F = F_init[:,0,0]
a = a_star[:,0,0]
c = c_star[:,0,0]

dFdW1  = finiteDiff_3D(F_init,0,1,hW1)





plt.plot(W1,F)
plt.savefig("F.pdf")
plt.close()

plt.plot(W1,a)
plt.savefig("a.pdf")
plt.close()

plt.plot(W1,c)
plt.savefig("c.pdf")
plt.close()