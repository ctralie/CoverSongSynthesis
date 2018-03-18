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
from NMFJoint import *

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

def getEuclideanErrorGPU(V, WH):
    """
    Return the Frobenius norm between V and W*H
    """
    diff = skcuda.misc.subtract(V, WH)
    diff = skcuda.misc.multiply(diff, diff)
    return skcuda.misc.sum(diff).get()

def multiplyConv2DWGradGPU(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for W using skcuda
    :param W: A TxNxK GPU array of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM GPU array of source activations for each submatrix of W\
            over F transpositions over M time
    :param V: An MxN GPU target array
    :param VLam: An MxN GPU estimate array
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    """
    thisV = V.copy()
    thisVLam = VLam.copy()
    thisH = H.copy()    
    WNums = gpuarray.zeros(W.shape, np.float32)
    WDenoms = gpuarray.zeros(W.shape, np.float32)
    for f in range(H.shape[0]):
        if f > 0:
            z = gpuarray.zeros((f, V.shape[1]), np.float32)
            thisV[0:-f, :] = V[f::, :]
            thisV[-f::, :] = z
            thisVLam[0:-f, :] = VLam[f::, :]
            thisVLam[-f::, :] = z
        for t in range(W.shape[0]):
            if t > 0:
                thisH[f, :, t::] = H[f, :, 0:-t]
                thisH[f, :, 0:t] = gpuarray.zeros((H.shape[1], t), np.float32)
            linalg.add_dot(thisV, thisH[f, :, :], WNums[t, :, :], transb='T')
            linalg.add_dot(thisVLam, thisH[f, :, :], WDenoms[t, :, :], transb='T')
    if doDivision:
        return skcuda.misc.divide(WNums, WDenoms)
    else:
        return (WNums, WDenoms)

def multiplyConv2DHGradGPU(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for H using skcuda
    :param W: A TxNxK GPU array of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM GPU array of source activations for each submatrix of W\
            over F transpositions over M time
    :param V: An MxN GPU target array
    :param VLam: An MxN GPU estimate array
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    """
    thisV = V.copy()
    thisVLam = VLam.copy()
    thisW = W.copy()    
    HNums = gpuarray.zeros(H.shape, np.float32)
    HDenoms = gpuarray.zeros(H.shape, np.float32)
    for t in range(W.shape[0]):
        if t > 0:
            #thisV = shiftMatLRUD(V, dj=-t)
            z = gpuarray.zeros((V.shape[0], t), np.float32)
            thisV[:, 0:-t] = V[:, t::]
            thisV[:, -t::] = z
            thisVLam[:, 0:-t] = VLam[:, t::]
            thisVLam[:, -t::] = z
        for f in range(H.shape[0]):
            if f > 0:
                #thisW = shiftMatLRUD(W[t, :, :], di=f)
                thisW[t, f::, :] = W[t, 0:-f, :]
                thisW[t, 0:f, :] = gpuarray.zeros((f, W.shape[2]), np.float32)
            linalg.add_dot(thisW[t, :, :], thisV, HNums[f, :, :], transa='T')
            linalg.add_dot(thisW[t, :, :], thisVLam, HDenoms[f, :, :], transa='T')
    if doDivision:
        return skcuda.misc.divide(HNums, HDenoms)
    else:
        return (HNums, HDenoms)

def multiplyConv2DGPU(W, H, blockdim = 32, Verbose = False):
    M = W.shape[1]
    N = H.shape[2]
    K = W.shape[2]
    T = W.shape[0]
    F = H.shape[0]
    sharedmem = 4*((F+blockdim)*T+(T+blockdim)*F)
    if Verbose:
        print("Shared Memory: %g kB"%(sharedmem/1024.0))

    #Figure out how to loop when copying over memory
    TBlockRound = blockdim*np.ceil(T/float(blockdim))
    FBlockRound = blockdim*np.ceil(F/float(blockdim))
    TBlocks = np.array(TBlockRound/blockdim, dtype=np.int32)
    FBlocks = np.array(FBlockRound/blockdim, dtype=np.int32)
    if Verbose:
        print("TBlocks = %i, FBlocks = %i"%(TBlocks, FBlocks))

    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    K = np.array(K, dtype=np.int32)
    T = np.array(T, dtype=np.int32)
    F = np.array(F, dtype=np.int32)
    
    GridDimM = int(np.ceil(1.0*M/blockdim))
    GridDimN = int(np.ceil(1.0*N/blockdim))
    Lam = gpuarray.zeros((M, N), np.float32)

    MatMulConv2D_(W, H, Lam, M, N, K, T, F, TBlocks, FBlocks, \
        block=(blockdim, blockdim, 1), \
        grid=(GridDimM, GridDimN, 1), shared=sharedmem )
    return Lam

def makeRandomGPUArray(M, N, L):
    return gpuarray.to_gpu(np.array(np.random.rand(M, N, L), dtype=np.float32))

def doNMF2DConvGPU(V, K, T, F, L, W = np.array([]), plotfn = None, \
    plotInterval = 60, plotFirst = False):
    """
    Implementing the Euclidean 2D NMF technique described in 
    "Nonnegative Matrix Factor 2-D Deconvolution
        for Blind Single Channel Source Separation"
    This is the GPU version, so matrices are converted to GPU arrays at
    the beginning of the function
    :param V: An N x M target matrix
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param F: Frequency extent of H matrices
    :param L: Number of iterations
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter) (assumed to take CPU arrays, \
        so conversion will be done in the function)
    :param plotInterval: At what interval to save plots of progress
    :returns (W, H): \
        W is an TxNxK CPU matrix of K sources over spatiotemporal spans NxT\
        H is a FxKxM CPU matrix of source activations for each submatrix of W\
            over F transpositions over M time
    """
    N = V.shape[0]
    M = V.shape[1]
    V = gpuarray.to_gpu(np.array(V, dtype=np.float32))
    WFixed = False
    if W.size == 0:
        W = makeRandomGPUArray(T, N, K)
    else:
        WFixed = True
        W = gpuarray.to_gpu(np.array(W, dtype=np.float32))
    H = makeRandomGPUArray(F, K, M)
    WH = multiplyConv2DGPU(W, H)
    errs = [getEuclideanErrorGPU(V, WH)]
    if plotfn:
        res=4
        pK = K
        plt.figure(figsize=((4+pK)*res, res))
        if plotFirst:
            plotfn(V.get(), W.get(), H.get(), 0, errs) 
            plt.savefig("NMF2DConv_%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))
        tic = time.time()
        #Step 1: Update Ws
        if not WFixed:
            VLam = multiplyConv2DGPU(W, H)
            WFac = multiplyConv2DWGradGPU(W, H, V, VLam)
            W = skcuda.misc.multiply(W, WFac)

        #Step 2: Update Hs
        VLam = multiplyConv2DGPU(W, H)
        HFac = multiplyConv2DHGradGPU(W, H, V, VLam)
        H = skcuda.misc.multiply(H, HFac)
        WH = multiplyConv2DGPU(W, H)
        errs.append(getEuclideanErrorGPU(V, WH))
        if plotfn and ((l+1) == L or (l+1)%plotInterval == 0):
            plt.clf()
            plotfn(V.get(), W.get(), H.get(), l+1, errs)
            plt.savefig("NMF2DConv_%i.png"%(l+1), bbox_inches = 'tight')
        print("Elapsed Time: %.3g"%(time.time()-tic))
    return (W.get(), H.get())

def doNMF2DConvJoint3WayGPU(A, Ap, B, K, T, F, L, plotfn = None, prefix = "", \
        plotInterval = 60, plotFirst = False):
    """
    A version of 2DNMF that jointly solves for W1, H1, W2, and H2 that
    satisfy W1*H1 ~= A, W2*H1 ~= Ap, and W2*H1 ~= B
    :param A: An M x N1 matrix for song A
    :param Ap: An M x N1 matrix for song A'
    :param B: An M x N2 matrix for song B
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param F: Frequency extent of H matrices
    :param L: Number of iterations
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns (W1, W2, H1, H2): \
        W1, W2 are an MxKxT matrices of K sources over spatiotemporal spans MxT\
        H1 is a KxN1xF matrix of source activations for the A, A' pair
        H2 is a KxN2xF matrix of source activations for B
    """
    if not (A.shape[1] == Ap.shape[1]):
        print("Error: A and A' should have same number of frames")
        return None
    M = A.shape[0]
    N1 = A.shape[1]
    N2 = B.shape[1]
    W1 = makeRandomGPUArray(T, M, K)
    W2 = makeRandomGPUArray(T, M, K)
    H1 = makeRandomGPUArray(F, K, N1)
    H2 = makeRandomGPUArray(F, K, N2)

    #Convert to GPU Arrays
    A = gpuarray.to_gpu(np.array(A, dtype=np.float32))
    Ap = gpuarray.to_gpu(np.array(Ap, dtype=np.float32))
    B = gpuarray.to_gpu(np.array(B, dtype=np.float32))

    errs = [getJoint3WayError(A, Ap, B, W1, W2, H1, H2, getEuclideanErrorGPU, multiplyConv2DGPU)]
    if plotfn:
        res=4
        plt.figure(figsize=((8+2*K)*res*1.2, 2*res))
        if plotFirst:
            plotfn(A.get(), Ap.get(), B.get(), W1.get(), W2.get(), H1.get(), H2.get(), 0, errs)
            pre = "%sNMFJoint3WayIter%i"%(prefix, 0)
            plt.savefig("%s/NMF2DConvJoint_%i.png"%(pre, 0), bbox_inches = 'tight')
    for l in range(L):
        print("Joint 2DNMF iteration %i of %i"%(l+1, L))
        tic = time.time()
        #Step 1: Update Ws
        Lam11 = multiplyConv2DGPU(W1, H1, Verbose = True)
        Lam12 = multiplyConv2DGPU(W1, H2, Verbose = True)
        Lam21 = multiplyConv2DGPU(W2, H1, Verbose = True)

        #Update W1
        (N11, D11) = multiplyConv2DWGradGPU(W1, H1, A, Lam11, doDivision = False)
        (N12, D12) = multiplyConv2DWGradGPU(W1, H2, B, Lam12, doDivision = False)
        Num = skcuda.misc.add(N11, N12)
        Denom = skcuda.misc.add(D11, D12)
        Fac = skcuda.misc.divide(Num, Denom)
        W1 = skcuda.misc.multiply(W1, Fac)
        #Update W2
        Fac = multiplyConv2DWGradGPU(W2, H1, Ap, Lam21, doDivision = True)
        W2 = skcuda.misc.multiply(W2, Fac)

        #Step 2: Update Hs
        #Update H1
        Lam11 = multiplyConv2DGPU(W1, H1)
        Lam12 = multiplyConv2DGPU(W1, H2)
        Lam21 = multiplyConv2DGPU(W2, H1)
        (N11, D11) = multiplyConv2DHGradGPU(W1, H1, A, Lam11, doDivision = False)
        (N12, D12) = multiplyConv2DHGradGPU(W2, H1, Ap, Lam21, doDivision = False)
        Num = skcuda.misc.add(N11, N12)
        Denom = skcuda.misc.add(D11, D12)
        Fac = skcuda.misc.divide(Num, Denom)
        H1 = skcuda.misc.multiply(H1, Fac)
        #Update H2
        Fac = multiplyConv2DHGradGPU(W1, H2, B, Lam12, doDivision = True)
        H2 = skcuda.misc.multiply(H2, Fac)

        errs.append(getJoint3WayError(A, Ap, B, W1, W2, H1, H2, getEuclideanErrorGPU, multiplyConv2DGPU))
        print("Elapsed Time: %.3g"%(time.time()-tic))
        if plotfn and ((l+1) == L or (l+1)%plotInterval == 0):
            plt.clf()
            plotfn(A.get(), Ap.get(), B.get(), W1.get(), W2.get(), H1.get(), H2.get(), l+1, errs)
            pre = "%sNMFJoint3WayIter%i"%(prefix, l+1)
            plt.savefig("%s/NMF2DConvJoint_%i.png"%(pre, l+1), bbox_inches = 'tight')
    return (W1.get(), W2.get(), H1.get(), H2.get())

def testNMF2DMultiplyGPU():
    initParallelAlgorithms()
    np.random.seed(100)
    M = 1025
    K = 10
    T = 12
    F = 40
    N = 20000
    W = np.random.rand(T, M, K)
    H = np.random.rand(F, K, N)

    tic = time.time()
    LamGT = multiplyConv2D(W, H)
    cputime = time.time()-tic
    print("Elapsed Time CPU: %.3g"%cputime)

    tic = time.time()
    WGPU = gpuarray.to_gpu(np.array(W, dtype=np.float32))
    HGPU = gpuarray.to_gpu(np.array(H, dtype=np.float32))
    Lam = multiplyConv2DGPU(WGPU, HGPU, blockdim = 32, Verbose = True)
    Lam = Lam.get()
    print("AllClose: ", np.allclose(Lam, LamGT))
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
    N = 10000
    V = np.random.rand(M, N)
    W = np.random.rand(T, M, K)
    H = np.random.rand(F, K, N)
    VLam = multiplyConv2D(W, H)

    sharedmem = 4*((blockdim+F)*blockdim*2+(T+blockdim)*F)
    print("Shared Memory: %g kB"%(sharedmem/1024.0))

    tic = time.time()
    HFacGT = multiplyConv2DHGrad(W, H, V, VLam)
    cputime = time.time()-tic
    print("Elapsed Time CPU: %.3g"%cputime)

    tic = time.time()
    WGPU = gpuarray.to_gpu(np.array(W, dtype=np.float32))
    HGPU = gpuarray.to_gpu(np.array(H, dtype=np.float32))
    VGPU = gpuarray.to_gpu(np.array(V, dtype=np.float32))
    VLamGPU = gpuarray.to_gpu(np.array(VLam, dtype=np.float32))

    HFac = multiplyConv2DHGradGPU(WGPU, HGPU, VGPU, VLamGPU)
    HFac = HFac.get()
    gputime = time.time()-tic
    print("Elapsed Time GPU: %.3g"%gputime)
    print("Speedup Ratio: %.3g"%(cputime/gputime))
    plt.figure(figsize=(16, 4))
    print(HFac.shape)
    print("AllClose: ", np.allclose(HFacGT, HFac))
    plot3Diff(HFacGT[:, 0, :], HFac[:, 0, :])
    plt.show()

if __name__ == '__main__':
    #testNMF2DMultiplyGPU()
    testNMF2DWGradientGPU()