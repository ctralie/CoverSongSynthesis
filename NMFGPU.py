import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath
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
MatMulConv2DWGrad_ = None

def initParallelAlgorithms():
    """
    Compile all of the parallel algorithms
    """
    global MatMulNaive_
    global MatMulConv2D_
    global MatMulConv2DWGrad_
    s = getResourceString("MatMul.cu")
    mod = SourceModule(s)
    MatMulNaive_ = mod.get_function("MatMulNaive")
    MatMulConv2D_ = mod.get_function("MatMulConv2D")
    MatMulConv2DWGrad_ = mod.get_function("MatMulConv2DWGrad")

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

def multiplyConv2DDebug(W, H):
    """
    Perform a convolutive matrix multiplication in time and frequency
    Debugging Version: Change this to only look at W or H at a time
    """
    Lam = np.zeros((W.shape[0], H.shape[1]), dtype=W.dtype)
    #Hf = np.ones((H.shape[0], H.shape[1]))
    Wt = np.ones((W.shape[0], W.shape[1]))
    for t in range(W.shape[2]):
        for f in range(H.shape[2]):
            #Wt = np.array(W[:, :, t])
            #Wt = shiftMatLRUD(Wt, di=f)
            Hf = np.array(H[:, :, f])
            Hf = shiftMatLRUD(Hf, dj=t)
            Lam += Wt.dot(Hf)
    return Lam

def testNMF2DMultiplyGPU():
    initParallelAlgorithms()
    np.random.seed(100)
    blockdim = 32
    M = 1025
    K = 10
    T = 40
    F = 40
    N = 1000
    W = np.random.randn(M, K, T)
    H = np.random.randn(K, N, F)

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
    M = 1025
    K = 10
    T = 20
    F = 40
    N = 2000
    V = np.random.randn(M, N)
    W = np.random.randn(M, K, T)
    H = np.random.randn(K, N, F)
    VLam = multiplyConv2D(W, H)

    sharedmem = 4*((blockdim+F)*blockdim*2+(T+blockdim)*F)
    print("Shared Memory: %g kB"%(sharedmem/1024.0))

    tic = time.time()
    Fac = multiplyConv2DWGrad(W, H, V, VLam)
    WNextGT = Fac*W
    cputime = time.time()-tic
    print("Elapsed Time CPU: %.3g"%cputime)

    tic = time.time()
    WGPU = gpuarray.to_gpu(np.array(W, dtype=np.float32))
    HGPU = gpuarray.to_gpu(np.array(H, dtype=np.float32))
    VGPU = gpuarray.to_gpu(np.array(V, dtype=np.float32))
    VLamGPU = gpuarray.to_gpu(np.array(VLam, dtype=np.float32))

    #Figure out how to loop when copying over memory
    FBlockRound = blockdim*np.ceil(F/float(blockdim))
    JBlocksRound = blockdim*np.ceil(N/float(blockdim))
    FBlocks = np.array(FBlockRound/blockdim, dtype=np.int32)
    JBlocks = np.array(JBlocksRound/blockdim, dtype=np.int32)

    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    K = np.array(K, dtype=np.int32)
    T = np.array(T, dtype=np.int32)
    F = np.array(F, dtype=np.int32)
    
    GridDimM = int(np.ceil(1.0*M/blockdim))
    GridDimT = int(np.ceil(1.0*T/blockdim))
    print("FBlocks = %i, JBlocks = %i"%(FBlocks, JBlocks))
    MatMulConv2DWGrad_(WGPU, HGPU, VGPU, VLamGPU, M, N, K, T, F, \
        FBlocks, JBlocks, block=(blockdim, blockdim, 1), \
        grid=(GridDimM, int(K), GridDimT), shared=sharedmem )
    
    WNext = WGPU.get()
    gputime = time.time()-tic
    print("Elapsed Time GPU: %.3g"%gputime)
    print("Speedup Ratio: %.3g"%(cputime/gputime))
    #plt.figure(figsize=(16, 4))
    #plot3Diff(WNextGT, WNext)
    #plt.show()

if __name__ == '__main__':
    testNMF2DMultiplyGPU()
    #testNMF2DWGradientGPU()