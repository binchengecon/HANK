from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import time
import petsclinearsystem
from petsc4py import PETSc
import petsc4py
import os
import sys
import numpy as np
sys.stdout.flush()
petsc4py.init(sys.argv)
reporterror = True

r = 0.1
sigma = 1.0


def finiteDiff_3D(data, dim, order, dlt, DBC0=0, DBC1 = -1, NBC1 =-2, cap=None):
    """
    F'(1)=-2 F(1)=-1 
             F(0)=0
    """

    # compute the central difference derivatives for given input and dimensions
    res1 = np.zeros(data.shape)
    res2 = np.zeros(data.shape)

    data[0,:,:]=DBC0
    data[-1,:,:]=DBC1

    if dim == 0:                  # to first dimension

        res1[1:-1, :, :] = (1 / (2 * dlt)) * \
            (data[2:, :, :] - data[:-2, :, :])
        res1[0,:,:] = (1/dlt)*(data[1,:,:]-data[0,:,:])
        res1[-1,:,:] = NBC1

        res2[1:-1, :, :] = (1 / (2 * dlt)) *  (res1[2:, :, :] - res1[:-2, :, :])
        res2[0, :, :] = (1/dlt)*(res1[1,:,:]-res1[0,:,:])
        res2[-1, :, :] = (1/dlt)*(res1[-1,:,:]-res1[-2,:,:])

    elif dim == 1:                # to second dimension

        res1[:, 1:-1, :] = (1 / (2 * dlt)) * \
            (data[:, 2:, :] - data[:, :-2, :])
        res1[:,0,:] = (1/dlt)*(data[:,1,:]-data[:,0,:])
        res1[:,-1,:] = (1/dlt)*(data[:,-1,:]-data[:,-2,:])

        res2[:, 1:-1, :] = (1 / (2 * dlt)) *  (res1[ :,2:, :] - res1[ :,:-2, :])
        res2[:,0,  :] = (1/dlt)*(res1[:,1,:]-res1[:,0,:])
        res2[:,-1, :] = (1/dlt)*(res1[:,-1,:]-res1[:,-2,:])

    elif dim == 2:                # to third dimension

        res1[:, :, 1:-1] = (1 / (2 * dlt)) * \
            (data[:, :, 2:] - data[:, :, :-2])
        res1[:,:,0] = (1/dlt)*(data[:,:,1]-data[:,:,0])
        res1[:,:,-1] = (1/dlt)*(data[:,:,-1]-data[:,:,-2])

        res2[:, :, 1:-1] = (1 / (2 * dlt)) *  (res1[ :, :,2:] - res1[ :, :,:-2])
        res2[:,  :,0] = (1/dlt)*(res1[:,:,1]-res1[:,:,0])
        res2[:, :,-1] = (1/dlt)*(res1[:,:,-1]-res1[:,:,-2])
    return res1, res2

def h(a):
    return a**2/2.0+0.4*a


def gamma(a):
    return a+0.4


def u(c):
    return np.sqrt(c)


def F0(c):
    return -c**2


W1_min = 0.0
W1_max = 1.0
hW1 = 0.005
W1 = np.arange(W1_min, W1_max+hW1, hW1)
nW1 = len(W1)

W1_short = W1[1:-1]
nW1_short = len(W1_short)

W2_min = 0.0
W2_max = 1.0
hW2 = 0.5
W2 = np.arange(W2_min, W2_max+hW2, hW2)
nW2 = len(W2)

W3_min = 0.0
W3_max = 1.0
hW3 = 0.5
W3 = np.arange(W3_min, W3_max+hW3, hW3)
nW3 = len(W3)

(W1_mat, W2_mat, W3_mat) = np.meshgrid(W1, W2, W3, indexing='ij')
stateSpace = np.hstack([W1_mat.reshape(-1, 1, order='F'),
                       W2_mat.reshape(-1, 1, order='F'), W3_mat.reshape(-1, 1, order='F')])

W1_mat_short = W1_mat[1:-1,:,:]

W1_mat_1d = W1_mat.ravel(order='F')
W1_mat_short_mat_1d = W1_mat_short.ravel(order='F')
W2_mat_1d = W2_mat.ravel(order='F')
W3_mat_1d = W3_mat.ravel(order='F')

lowerLims = np.array([W1.min(), W2.min(), W3.min()], dtype=np.float64)
upperLims = np.array([W1.max(), W2.max(), W3.max()], dtype=np.float64)

lowerLims_short = np.array([W1_short.min(), W2.min(), W3.min()], dtype=np.float64)
upperLims_short = np.array([W1_short.max(), W2.max(), W3.max()], dtype=np.float64)


print("Grid dimension: [{}, {}, {}]\n".format(nW1, nW2, nW3))
print("Grid step: [{}, {}, {}]\n".format(hW1, hW2, hW3))

F_init = -W1_mat**2
# F_init = -W1_mat**2

# a_star = np.zeros(W1_mat.shape)
# c_star = np.zeros(W1_mat.shape)

a_star = np.zeros(W1_mat_short.shape)
c_star = np.zeros(W1_mat_short.shape)


dVec = np.array([hW1, hW2, hW3])
increVec = np.array([1, nW1_short, nW1_short*nW2], dtype=np.int32)

petsc_mat = PETSc.Mat().create()
petsc_mat.setType('aij')
petsc_mat.setSizes([nW1_short * nW2 * nW3, nW1_short * nW2 * nW3])
petsc_mat.setPreallocationNNZ(13)
petsc_mat.setUp()
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setType('bcgs')
ksp.getPC().setType('ilu')
ksp.setFromOptions()

FC_Err = 1
epoch = 0
max_iter = 10
tol = 1e-8
# fraction = 0.1
# epsilon = 0.01
fraction = 0.01
epsilon = 0.5

while FC_Err > tol and epoch < max_iter:
    start_eps = time.time()
    F_init[0,:,:]=0
    F_init[-1,:,:]=-1
    F_init_short = F_init[1:-1,:,:]
    dFdW1,ddFddW1 = finiteDiff_3D(F_init, 0, 1, hW1)

    dFdW1_short= dFdW1[1:-1,:,:]
    ddFddW1_short = ddFddW1[1:-1,:,:]
    # dFdW1[0,:,:][dFdW1[0,:,:]<=1e-16]=1e-16

    # dW1[dW1 <= 1e-16] = 1e-16
    # dK = dW1
    dFdW2,ddFddW2 = finiteDiff_3D(F_init, 1, 1, hW2)
    # dY = dW2
    dFdW3,ddFddW3 = finiteDiff_3D(F_init, 2, 1, hW3)
    # dW3[dW3 <= 1e-16] = 1e-16
    # dL = dW3
    # second order
    #  = finiteDiff_3D(F_init, 0, 2, hW1)
    # ddFddW2 = finiteDiff_3D(F_init, 1, 2, hW2)
    # # ddY = ddW2
    # ddFddW3 = finiteDiff_3D(F_init, 2, 2, hW3)

    # need to change the control optimizatio completely due to corner solution of c

    # if np.any(dFdW1+ddFddW1 * r * sigma**2 >= 0):
    #     print("warning\n")

    a_den = dFdW1_short+ddFddW1_short * r * sigma**2
    print(a_den.shape)
    # a_den[a_den >= -1e-1] = -1e-1

    a_new = -1/a_den-0.4

    a_new[0, :, :] =1e-3

    a_new[a_new <= 1e-3] = 1e-3

    c_new = (dFdW1_short/2)**2
    c_new[ddFddW1_short >= 0] = 0

    # c_new[c_new<=1e-16] = 1e-16

    a = a_new * fraction + a_star*(1-fraction)
    c = c_new * fraction + c_star*(1-fraction)
    A = -r*np.ones(W1_mat_short.shape)
    B_1 = r*(W1_mat_short-u(c)+h(a))
    B_2 = np.zeros(W1_mat_short.shape)
    B_3 = np.zeros(W1_mat_short.shape)
    C_1 = r**2*sigma**2*gamma(a)**2/2
    C_2 = np.zeros(W1_mat_short.shape)
    C_3 = np.zeros(W1_mat_short.shape)
    D = r*(a-c)

    start_ksp = time.time()

    A_1d = A.ravel(order='F')
    C_1_1d = C_1.ravel(order='F')
    C_2_1d = C_2.ravel(order='F')
    C_3_1d = C_3.ravel(order='F')
    B_1_1d = B_1.ravel(order='F')
    B_2_1d = B_2.ravel(order='F')
    B_3_1d = B_3.ravel(order='F')
    D_1d = D.ravel(order='F')
    petsclinearsystem.formLinearSystem(W1_mat_1d, W2_mat_1d, W3_mat_1d, A_1d, B_1_1d, B_2_1d,
                                       B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims_short, upperLims_short, dVec, increVec, petsc_mat)
    F_init_short_1d = F_init_short.ravel(order='F')
    b = F_init_short_1d + D_1d * epsilon
    petsc_rhs = PETSc.Vec().createWithArray(b)
    x = petsc_mat.createVecRight()

    # create linear solver
    start_ksp = time.time()
    ksp.setOperators(petsc_mat)
    ksp.setTolerances(rtol=tol)
    ksp.solve(petsc_rhs, x)
    petsc_rhs.destroy()
    x.destroy()
    out_comp = np.array(ksp.getSolution()).reshape(A.shape, order="F")
    end_ksp = time.time()
    num_iter = ksp.getIterationNumber()

    PDE_rhs = A * F_init_short + B_1 * dFdW1_short + C_1 * ddFddW1_short + D
    PDE_Err = np.max(abs(PDE_rhs))
    FC_Err = np.max(abs((out_comp - F_init_short) / epsilon))

    F_init[1:-1,:,:] = out_comp
    F_init[0, :, :] = 0
    F_init[-1, :, :] = -1
    # F_init[-2,:,:] = F0(W1[-2])

    a_star = a
    c_star = c
    epoch += 1

    print("petsc total: {:.3f}s".format(end_ksp - start_ksp))
    print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(
        ksp.getResidualNorm(), ksp.getIterationNumber()))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(
        epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_eps))


res = {
    "F_init": F_init,
    "a_star": a_star,
    "c_star": c_star,
    "FC_Err": FC_Err,
    "W1": W1,
    "W2": W2,
    "W3": W3,
}

Data_Dir = "./Python/data/"
os.makedirs(Data_Dir, exist_ok=True)

with open(Data_Dir + "model_result4_part", "wb") as f:
    pickle.dump(res, f)


F = F_init[:, 0, 0]
a = a_star[:, 0, 0]
c = c_star[:, 0, 0]


font = {'family': 'monospace',

        'weight': 'bold',

        'size': 18}


plt.rc('font', **font)  # pass in the font dict as kwargs


figwidth = 10


fig, axs = plt.subplot_mosaic(

    [["left column", "right top"],
     ["left column", "right mid"],
     ["left column", "right down"]], figsize=(4 * figwidth, 2 * figwidth)

)


axs["left column"].plot(W1, F)
axs["left column"].plot(W1, -W1**2)
axs["left column"].set_title("Profit")
axs["left column"].grid(linestyle=':')


axs["right top"].plot(W1, a)
axs["right top"].set_title("Effort a(W)")
axs["right top"].grid(linestyle=':')


axs["right mid"].plot(W1, c)
axs["right mid"].set_title("Consumption c(W)")
axs["right mid"].grid(linestyle=':')

B_W = r*(W1-c**(1/2)+a**2/2+2*a/5)

axs["right down"].plot(W1, B_W)
axs["right down"].set_title("Drift of W")
axs["right down"].grid(linestyle=':')

pdf_pages = PdfPages(f"./Python/Result4_{max_iter}.pdf")
pdf_pages.savefig(fig)
plt.close()
pdf_pages.close()
