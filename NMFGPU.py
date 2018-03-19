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

#Global handles to CUDA kernels
MatMulNaive_ = None
MatMulConv2D_ = None
ZerosToOnes_ = None
TileWDenom_ = None
TileHDenom_ = None

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
    
    global ZerosToOnes_
    global TileWDenom_
    global TileHDenom_
    s = getResourceString("OtherUtils.cu")
    mod = SourceModule(s)
    ZerosToOnes_ = mod.get_function("ZerosToOnes")
    TileWDenom_ = mod.get_function("TileWDenom")
    TileHDenom_ = mod.get_function("TileHDenom")

    linalg.init()
    skcuda.misc.init()

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

def ZerosToOnes(V, eps = 0.0):
    """
    :param V: An MxN GPU array
    """
    blockdim = 32
    M = V.shape[0]
    N = V.shape[1]
    GridDimM = int(np.ceil(1.0*M/blockdim))
    GridDimN = int(np.ceil(1.0*N/blockdim))
    M = np.array(M, dtype = np.int32)
    N = np.array(N, dtype = np.int32)
    eps = np.array(eps, dtype = np.float32)
    ZerosToOnes_(V, M, N, eps, block=(blockdim, blockdim, 1), \
        grid=(GridDimM, GridDimN))

def getEuclideanErrorGPU(V, WH):
    """
    Return the Frobenius norm between V and W*H
    """
    diff = skcuda.misc.subtract(V, WH)
    diff = skcuda.misc.multiply(diff, diff)
    return skcuda.misc.sum(diff).get()

def getKLErrorGPU(V, WH, eps = 1e-10):
    """
    Return the Kullback-Liebler diverges between V and W*H
    """
    denom = WH.copy()
    ZerosToOnes(denom) #Prevent divide by zero
    arg = skcuda.misc.divide(V, denom)
    ZerosToOnes(arg, eps)
    res = skcuda.misc.multiply(V, pycuda.cumath.log(arg))
    res = skcuda.misc.subtract(res, V)
    res = skcuda.misc.add(res, WH)
    res = skcuda.misc.sum(res).get()
    print(res)
    return res

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

def TileWDenom(WDenomIn, M):
    """
    :param WDenomIn: A T x 1 x K array that needs to be tiled
    :param M: Dimension of tile axis
    :returns: WDenomOut: A T x M x K tiled array
    """
    blockdim = 32
    T = WDenomIn.shape[0]
    K = WDenomIn.shape[2]
    GridDimT = int(np.ceil(1.0*T/blockdim))
    GridDimK = int(np.ceil(1.0*K/blockdim))
    T = np.array(T, dtype = np.int32)
    M = np.array(M, dtype = np.int32)
    K = np.array(K, dtype = np.int32)
    WDenomOut = gpuarray.zeros((T, M, K), np.float32)
    TileWDenom_(WDenomIn, WDenomOut, T, M, K, block=(blockdim, blockdim, 1), \
        grid=(GridDimT, GridDimK))
    return WDenomOut


def multiplyConv2DWGradKLGPU(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for W under the
    Kullback-Liebler divergence, using skcuda
    :param W: A TxNxK matrix of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    :param VLam: Convolutional WH multiplication
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    :returns Ratio: A TxNkX matrix of multiplicative updates for W\
        or (RatioNum, RatioDenom) if doDivision = False
    """
    WNums = gpuarray.zeros(W.shape, np.float32)
    WDenoms = gpuarray.zeros((W.shape[0], W.shape[2]), np.float32)
    thisVLam = VLam.copy()
    ZerosToOnes(thisVLam)
    VLamQuot = skcuda.misc.divide(V, thisVLam)
    thisVLamQuot = VLamQuot.copy()
    thisH = H.copy()
    for f in range(H.shape[0]):
        if f > 0:
            thisVLamQuot[0:-f, :] = VLamQuot[f::, :]
            thisVLamQuot[-f::, :] = gpuarray.zeros((f, V.shape[1]), np.float32)
        for t in range(W.shape[0]):
            if t > 0:
                thisH[f, :, t::] = H[f, :, 0:-t]
                thisH[f, :, 0:t] = gpuarray.zeros((H.shape[1], t), np.float32)
            linalg.add_dot(thisVLamQuot, thisH[f, :, :], WNums[t, :, :], transb='T')
            WDenoms[t, :] = skcuda.misc.add(WDenoms[t, :], skcuda.misc.sum(thisH[f, :, :], 1))
    WDenoms = TileWDenom(WDenoms[:, None, :], W.shape[1])
    if doDivision:
        return skcuda.misc.divide(WNums, WDenoms)
    else:
        return (WNums, WDenoms)

def TileHDenom(HDenomIn, N):
    """
    :param HDenomIn: A F x K x 1 array that needs to be tiled
    :param N: Dimension of tile axis
    :returns: HDenomOut: A F x K x N tiled array
    """
    blockdim = 32
    F = HDenomIn.shape[0]
    K = HDenomIn.shape[1]
    GridDimF = int(np.ceil(1.0*F/blockdim))
    GridDimK = int(np.ceil(1.0*K/blockdim))
    F = np.array(F, dtype = np.int32)
    K = np.array(K, dtype = np.int32)
    N = np.array(N, dtype = np.int32)
    HDenomOut = gpuarray.zeros((F, K, N), np.float32)
    TileHDenom_(HDenomIn, HDenomOut, F, K, N, block=(blockdim, blockdim, 1), \
        grid=(GridDimF, GridDimK))
    return HDenomOut

def multiplyConv2DHGradKLGPU(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for H
    under the Kullback-Liebler divergence, using skcuda
    :param W: A TxNxK matrix of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    :param VLam: Convolutional WH multiplication
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    :returns Ratio: A FxKxM matrix of multiplicative updates for H\
        or (RatioNum, RatioDenom) if doDivision = False
    """
    HNums = gpuarray.zeros(H.shape, np.float32)
    HDenoms = gpuarray.zeros((H.shape[0], H.shape[1]), np.float32)
    thisVLam = VLam.copy()
    ZerosToOnes(thisVLam)
    VLamQuot = skcuda.misc.divide(V, thisVLam)
    thisVLamQuot = VLamQuot.copy()
    thisW = W.copy()
    for t in range(W.shape[0]):
        if t > 0:
            z = gpuarray.zeros((V.shape[0], t), np.float32)
            thisVLamQuot[:, 0:-t] = VLamQuot[:, t::]
            thisVLamQuot[:, -t::] = z
        for f in range(H.shape[0]):
            if f > 0:
                thisW[t, f::, :] = W[t, 0:-f, :]
                thisW[t, 0:f, :] = gpuarray.zeros((f, W.shape[2]), np.float32)
            linalg.add_dot(thisW[t, :, :], thisVLamQuot, HNums[f, :, :], transa='T')
            HDenoms[f, :] = skcuda.misc.add(HDenoms[f, :], skcuda.misc.sum(thisW[t, :, :], 0))
    HDenoms = TileHDenom(HDenoms[:, :, None], H.shape[2])
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

def doNMF2DConvGPU(V, K, T, F, L, W = np.array([]), doKL = False, plotfn = None, \
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
    :param doKL: Whether to do Kullback-Leibler divergence.  If false, do Euclidean
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
    errfn = getEuclideanErrorGPU
    WGradfn = multiplyConv2DWGradGPU
    HGradfn = multiplyConv2DHGradGPU
    if doKL:
        errfn = getKLErrorGPU
        WGradfn = multiplyConv2DWGradKLGPU
        HGradfn = multiplyConv2DHGradKLGPU
    errs = [errfn(V, WH)]
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
            WFac = WGradfn(W, H, V, VLam)
            W = skcuda.misc.multiply(W, WFac)

        #Step 2: Update Hs
        VLam = multiplyConv2DGPU(W, H)
        HFac = HGradfn(W, H, V, VLam)
        H = skcuda.misc.multiply(H, HFac)
        WH = multiplyConv2DGPU(W, H)
        errs.append(errfn(V, WH))
        if plotfn and ((l+1) == L or (l+1)%plotInterval == 0):
            plt.clf()
            plotfn(V.get(), W.get(), H.get(), l+1, errs)
            plt.savefig("NMF2DConv_%i.png"%(l+1), bbox_inches = 'tight')
        print("Elapsed Time: %.3g"%(time.time()-tic))
    return (W.get(), H.get())


def doNMF2DConvJointGPU(A, Ap, K, T, F, L, W = np.array([]), doKL = False, plotfn = None, \
    plotInterval = 60, plotFirst = False, prefix = "", errfn = getEuclideanErrorGPU):
    """
    Do a joint version of 2DNMF solving for W1, W2, and H, where 
    A ~= W1*H and Ap ~= W2*H
    This is the GPU version, so matrices are converted to GPU arrays at
    the beginning of the function
    :param A: An N x M target matrix for dataset 1
    :param Ap: An N x M target matrix for dataset 2
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param F: Frequency extent of H matrices
    :param L: Number of iterations
    :param doKL: Whether to do Kullback-Leibler divergence.  If false, do Euclidean
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns (W1, W2, H): \
        W1 is an TxNxK matrix of K A sources over spatiotemporal spans NxT\
        W2 is an TxNxK matrix of K Ap sources over spatiotemporal spans NxT\
        H is a FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    """
    N = A.shape[0]
    M = A.shape[1]
    A = gpuarray.to_gpu(np.array(A, dtype=np.float32))
    Ap = gpuarray.to_gpu(np.array(Ap, dtype=np.float32))
    W1 = makeRandomGPUArray(T, N, K)
    W2 = makeRandomGPUArray(T, N, K)
    H = makeRandomGPUArray(F, K, M)

    errfn = getEuclideanErrorGPU
    WGradfn = multiplyConv2DWGradGPU
    HGradfn = multiplyConv2DHGradGPU
    if doKL:
        errfn = getKLErrorGPU
        WGradfn = multiplyConv2DWGradKLGPU
        HGradfn = multiplyConv2DHGradKLGPU

    errs = [[errfn(A, multiplyConv2DGPU(W1, H)), errfn(Ap, multiplyConv2DGPU(W2, H))]]
    if plotfn:
        res=4
        plt.figure(figsize=((2+K)*res, 2*res))
        pre = "%sNMF2DJointIter%i"%(prefix, 0)
        if not os.path.exists(pre):
            os.mkdir(pre)
        if plotFirst:
            plotfn(A.get(), Ap.get(), W1.get(), W2.get(), H.get(), 0, errs)
            plt.savefig("%s/NMF2DConvJoint_%i.png"%(pre, 0), bbox_inches = 'tight')
    for l in range(L):
        print("Joint 2DNMF iteration %i of %i"%(l+1, L))
        tic = time.time()
        #Step 1: Update Ws
        ALam = multiplyConv2DGPU(W1, H)
        ApLam = multiplyConv2DGPU(W2, H)
        WFac = WGradfn(W1, H, A, ALam)
        W1 = skcuda.misc.multiply(W1, WFac)
        WFac = WGradfn(W2, H, Ap, ApLam)
        W2 = skcuda.misc.multiply(W2, WFac)

        #Step 2: Update Hs
        ALam = multiplyConv2DGPU(W1, H)
        ApLam = multiplyConv2DGPU(W2, H)
        (HNums1, HDenoms1)= HGradfn(W1, H, A, ALam, doDivision = False)
        (HNums2, HDenoms2)= HGradfn(W2, H, Ap, ApLam, doDivision = False)
        HNums = skcuda.misc.add(HNums1, HNums2)
        HDenoms = skcuda.misc.add(HDenoms1, HDenoms2)
        HFac = skcuda.misc.divide(HNums, HDenoms)
        H = skcuda.misc.multiply(H, HFac)

        #Plot results
        errs.append([errfn(A, multiplyConv2DGPU(W1, H)), errfn(Ap, multiplyConv2DGPU(W2, H))])
        if plotfn and ((l+1) == L or (l+1)%plotInterval == 0):
            plt.clf()
            plotfn(A.get(), Ap.get(), W1.get(), W2.get(), H.get(), l+1, errs)
            pre = "%sNMF2DJointIter%i"%(prefix, l+1)
            if not os.path.exists(pre):
                os.mkdir(pre)
            plt.savefig("%s/NMF2DConvJoint_%i.png"%(pre, l+1), bbox_inches = 'tight')
        print("Elapsed Time: %.3g"%(time.time()-tic))
    return (W1.get(), W2.get(), H.get())



def doNMF2DConvJoint3WayGPU(A, Ap, B, K, T, F, L, doKL = False, plotfn = None, \
         prefix = "", plotInterval = 60, plotFirst = False):
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
    :param doKL: Whether to do Kullback-Leibler divergence.  If false, do Euclidean
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

    errfn = getEuclideanErrorGPU
    WGradfn = multiplyConv2DWGradGPU
    HGradfn = multiplyConv2DHGradGPU
    if doKL:
        errfn = getKLErrorGPU
        WGradfn = multiplyConv2DWGradKLGPU
        HGradfn = multiplyConv2DHGradKLGPU

    errs = [getJoint3WayError(A, Ap, B, W1, W2, H1, H2, errfn, multiplyConv2DGPU)]
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
        Lam11 = multiplyConv2DGPU(W1, H1)
        Lam12 = multiplyConv2DGPU(W1, H2)
        Lam21 = multiplyConv2DGPU(W2, H1)

        #Update W1
        (N11, D11) = WGradfn(W1, H1, A, Lam11, doDivision = False)
        (N12, D12) = WGradfn(W1, H2, B, Lam12, doDivision = False)
        Num = skcuda.misc.add(N11, N12)
        Denom = skcuda.misc.add(D11, D12)
        Fac = skcuda.misc.divide(Num, Denom)
        W1 = skcuda.misc.multiply(W1, Fac)
        #Update W2
        Fac = WGradfn(W2, H1, Ap, Lam21, doDivision = True)
        W2 = skcuda.misc.multiply(W2, Fac)

        #Step 2: Update Hs
        #Update H1
        Lam11 = multiplyConv2DGPU(W1, H1)
        Lam12 = multiplyConv2DGPU(W1, H2)
        Lam21 = multiplyConv2DGPU(W2, H1)
        (N11, D11) = HGradfn(W1, H1, A, Lam11, doDivision = False)
        (N12, D12) = HGradfn(W2, H1, Ap, Lam21, doDivision = False)
        Num = skcuda.misc.add(N11, N12)
        Denom = skcuda.misc.add(D11, D12)
        Fac = skcuda.misc.divide(Num, Denom)
        H1 = skcuda.misc.multiply(H1, Fac)
        #Update H2
        Fac = HGradfn(W1, H2, B, Lam12, doDivision = True)
        H2 = skcuda.misc.multiply(H2, Fac)

        errs.append(getJoint3WayError(A, Ap, B, W1, W2, H1, H2, errfn, multiplyConv2DGPU))
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

def testKLGradient():
    initParallelAlgorithms()
    M = 1025
    K = 10
    T = 20
    F = 40
    N = 10000
    V = np.random.rand(M, N)
    W = np.random.rand(T, M, K)
    H = np.random.rand(F, K, N)

    WGPU = gpuarray.to_gpu(np.array(W, dtype=np.float32))
    HGPU = gpuarray.to_gpu(np.array(H, dtype=np.float32))
    VGPU = gpuarray.to_gpu(np.array(V, dtype=np.float32))
    VLamGPU = multiplyConv2DGPU(WGPU, HGPU)
    VLam = VLamGPU.get()

    ##Step 1: Test W Gradient
    #Test separating out numerators and denomenators
    tic = time.time()
    (WNums, WDenoms) = multiplyConv2DWGradKL(W, H, V, VLam, doDivision = False)
    print("Elapsed Time CPU: %.3g"%(time.time() - tic))
    tic = time.time()
    (WNumsGPU, WDenomsGPU) = multiplyConv2DWGradKLGPU(WGPU, HGPU, VGPU, VLamGPU, doDivision = False)    
    print("Elapsed Time GPU: %.3g"%(time.time() - tic))
    print("All Close W Nums: ", np.allclose(WNums, WNumsGPU.get()))

    #Test the ratio
    tic = time.time()
    WRatio = multiplyConv2DWGradKL(W, H, V, VLam)
    print("Elapsed Time CPU: %.3g"%(time.time() - tic))
    tic = time.time()
    WRatioGPU = multiplyConv2DWGradKLGPU(WGPU, HGPU, VGPU, VLamGPU) 
    print("Elapsed Time GPU: %.3g"%(time.time() - tic))
    print("All Close W Ratio: ", np.allclose(WRatio, WRatioGPU.get()))


    ##Step 2: Test H Gradient
    #Test separating out numerators and denomenators
    tic = time.time()
    (HNums, HDenoms) = multiplyConv2DHGradKL(W, H, V, VLam, doDivision = False)
    print("Elapsed Time CPU: %.3g"%(time.time() - tic))
    tic = time.time()
    (HNumsGPU, HDenomsGPU) = multiplyConv2DHGradKLGPU(WGPU, HGPU, VGPU, VLamGPU, doDivision = False)    
    print("Elapsed Time GPU: %.3g"%(time.time() - tic))
    print("All Close H Nums: ", np.allclose(HNums, HNumsGPU.get()))

    #Test the ratio
    tic = time.time()
    HRatio = multiplyConv2DHGradKL(W, H, V, VLam)
    print("Elapsed Time CPU: %.3g"%(time.time() - tic))
    tic = time.time()
    HRatioGPU = multiplyConv2DHGradKLGPU(WGPU, HGPU, VGPU, VLamGPU) 
    print("Elapsed Time GPU: %.3g"%(time.time() - tic))
    print("All Close H Ratio: ", np.allclose(HRatio, HRatioGPU.get()))



if __name__ == '__main__':
    #testNMF2DMultiplyGPU()
    #testNMF2DWGradientGPU()
    testKLGradient()