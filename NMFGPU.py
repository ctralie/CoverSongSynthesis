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

def testNMF2DMultiplyGPU():
    initParallelAlgorithms()
    np.random.seed(100)
    blockdim = 32
    M = 1025
    K = 10
    T = 40
    F = 40
    N = 10000
    W = np.random.randn(M, K, T)

    H = np.random.randn(K, N, F)
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
    MatMulConv2D_(WGPU, HGPU, Lam, M, N, K, T, F, TBlocks, FBlocks, \
        block=(blockdim, blockdim, 1), \
        grid=(GridDimM, GridDimN, 1), shared=4*((F+blockdim)*T+(T+blockdim)*F) )

    
    Lam = Lam.get()
    gputime = time.time()-tic
    print("Elapsed Time GPU: %.3g"%gputime)
    print("Speedup Ratio: %.3g"%(cputime/gputime))
    plt.figure(figsize=(16, 4))
    plot3Diff(LamGT, Lam)
    plt.savefig("fig.png", bbox_inches='tight')

if __name__ == '__main__':
    testNMF2DMultiplyGPU()