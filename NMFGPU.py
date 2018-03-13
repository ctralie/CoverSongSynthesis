import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath
import skcuda.misc
import skcuda.linalg as linalg
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import pkg_resources
from NMF import *

def getResourceString(filename):
    fin = open(filename)
    s = fin.read()
    fin.close()
    return s.decode('utf8')

MatMulNaive_ = None
MatMulConv2D_ = None

def initParallelAlgorithms():
    """
    Compile all of the parallel algorithms
    """
    global MatMulNaive_
    global MatMulConv2D_
    s = getResourceString("MatMul.cu")
    mod = SourceModule(s)
    MatMulNaive_ = mod.get_function("MatMulNaive")
    MatMulConv2D_ = mod.get_function("MatMulConv2D")
    linalg.init()

def plot3Diff(A, B, eps = None):
    """
    For debugging CPU vs GPU; plot a CPU result versus a GPU result
    along with a difference, with colorbars, to see if error is within
    numerical precision
    :param A: An MxN matrix from the CPU
    :param B: An MxN matrix from the GPU
    :param eps: Plot a black/white image with white pixels if the\
        difference is above eps
    """
    plt.subplot(131)
    plt.imshow(A, cmap = 'afmhot', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.title("CPU")
    plt.subplot(132)
    plt.imshow(B, cmap = 'afmhot', interpolation='none', aspect='auto')
    plt.title("GPU")
    plt.colorbar()
    plt.subplot(133)
    diff = A-B
    if eps:
        diff2 = np.ones(diff.shape)
        diff2[np.abs(diff) < eps] = 0
        diff = diff2    
    plt.imshow(diff, cmap = 'afmhot', interpolation='none', aspect='auto')
    plt.title("Diff")
    plt.colorbar()

def multiplyConv2DWGradDebug(W, H, V, VLam):
    """
    Compute the 2D convolutional multiplicative update
    for W
    :param W: A NxKxT matrix of K sources over spatiotemporal spans NxT\
    :param H: A KxMxF matrix of source activations for each submatrix of W\
            over F transpositions over M time
    """
    WNums = np.zeros(W.shape) #Numerator
    WDenoms = np.zeros(W.shape) #Denomenator
    for f in range(H.shape[0]):
        thisV = shiftMatLRUD(V, di=-f)
        thisVLam = shiftMatLRUD(VLam, di=-f)
        for t in range(W.shape[0]):
            thisH = shiftMatLRUD(H[f, :, :], dj=t)
            WNums[t, :, :] += thisV.dot(thisH.T)
            #WDenoms[t, :, :] += thisVLam.dot(thisH.T)
    return WNums

def multiplyConv2DWGradGPU(W, H, V, VLam):
    thisV = V.copy()
    thisVLam = VLam.copy()
    thisH = H.copy()    
    WNums = gpuarray.zeros(W.shape, np.float32)
    WDenoms = gpuarray.zeros(W.shape, np.float32)
    for f in range(H.shape[0]):
        if f > 0:
            thisV[0:-f, :] = V[f::, :]
            thisV[-f::, :] = gpuarray.zeros((f, V.shape[1]), np.float32)
        for t in range(W.shape[0]):
            if t > 0:
                thisH[f, :, t::] = H[f, :, 0:-t]
                thisH[f, :, 0:t] = gpuarray.zeros((H.shape[1], t), np.float32)
            linalg.add_dot(thisV, thisH[f, :, :], WNums[t, :, :], transb='T')
    return WNums


def testNMF2DMultiplyGPU():
    initParallelAlgorithms()
    np.random.seed(100)
    blockdim = 32
    M = 1025
    K = 10
    T = 12
    F = 40
    N = 1000
    W = np.random.rand(T, M, K)
    H = np.random.rand(F, K, N)

    sharedmem = 4*((F+blockdim)*T+(T+blockdim)*F)
    print("Shared Memory: %g kB"%(sharedmem/1024.0))

    tic = time.time()
    LamGT = multiplyConv2D(W, H)
    cputime = time.time()-tic
    print("Elapsed Time CPU: %.3g"%cputime)

    tic = time.time()
    WGPU = gpuarray.to_gpu(np.array(W, dtype=np.float32))
    HGPU = gpuarray.to_gpu(np.array(H, dtype=np.float32))
    Lam = np.zeros((M, N), dtype=np.float32)
    Lam = gpuarray.to_gpu(Lam)

    #Figure out how to loop when copying over memory
    TBlockRound = blockdim*np.ceil(T/float(blockdim))
    FBlockRound = blockdim*np.ceil(F/float(blockdim))
    TBlocks = np.array(TBlockRound/blockdim, dtype=np.int32)
    FBlocks = np.array(FBlockRound/blockdim, dtype=np.int32)

    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    K = np.array(K, dtype=np.int32)
    T = np.array(T, dtype=np.int32)
    F = np.array(F, dtype=np.int32)
    
    GridDimM = int(np.ceil(1.0*M/blockdim))
    GridDimN = int(np.ceil(1.0*N/blockdim))
    
    print("TBlocks = %i, FBlocks = %i"%(TBlocks, FBlocks))
    MatMulConv2D_(WGPU, HGPU, Lam, M, N, K, T, F, TBlocks, FBlocks, \
        block=(blockdim, blockdim, 1), \
        grid=(GridDimM, GridDimN, 1), shared=sharedmem )

    
    Lam = Lam.get()
    gputime = time.time()-tic
    print("Elapsed Time GPU: %.3g"%gputime)
    print("Speedup Ratio: %.3g"%(cputime/gputime))
    plt.figure(figsize=(16, 4))
    plot3Diff(LamGT, Lam)
    #plt.savefig("fig.png", bbox_inches='tight')
    plt.show()

def testNMF2DWGradientGPU():
    initParallelAlgorithms()
    np.random.seed(100)
    blockdim = 32
    M = 300
    K = 10
    T = 20
    F = 40
    N = 1000
    V = np.random.rand(M, N)
    W = np.random.rand(T, M, K)
    H = np.random.rand(F, K, N)
    VLam = multiplyConv2D(W, H)

    sharedmem = 4*((blockdim+F)*blockdim*2+(T+blockdim)*F)
    print("Shared Memory: %g kB"%(sharedmem/1024.0))

    tic = time.time()
    WNumsGT = multiplyConv2DWGradDebug(W, H, V, VLam)
    cputime = time.time()-tic
    print("Elapsed Time CPU: %.3g"%cputime)

    tic = time.time()
    WGPU = gpuarray.to_gpu(np.array(W, dtype=np.float32))
    HGPU = gpuarray.to_gpu(np.array(H, dtype=np.float32))
    VGPU = gpuarray.to_gpu(np.array(V, dtype=np.float32))
    VLamGPU = gpuarray.to_gpu(np.array(VLam, dtype=np.float32))

    WNums = multiplyConv2DWGradGPU(WGPU, HGPU, VGPU, VLamGPU)
    WNums = WNums.get()
    #WNums = multiplyConv2DWGradDebug2(W, H, V, VLam)
    gputime = time.time()-tic
    print("Elapsed Time GPU: %.3g"%gputime)
    print("Speedup Ratio: %.3g"%(cputime/gputime))
    plt.figure(figsize=(16, 4))
    plot3Diff(WNumsGT[:, :, 0], WNums[:, :, 0])
    plt.show()


if __name__ == '__main__':
    testNMF2DMultiplyGPU()
    #testNMF2DWGradientGPU()