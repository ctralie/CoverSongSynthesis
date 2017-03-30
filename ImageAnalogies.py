import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
from skimage.transform import pyramid_gaussian

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def readImage(filename):
    I = scipy.misc.imread(filename)
    I = np.array(I, dtype=np.float32)/255.0
    return I

def writeImage(I, filename):
    IRet = np.array(I*255.0, dtype=np.uint8)
    scipy.misc.imsave(filename, IRet)

def getPatches(I, dim):
    #http://stackoverflow.com/questions/13682604/slicing-a-numpy-image-array-into-blocks
    shape = np.array(I.shape*2)
    strides = np.array(I.strides*2)
    W = np.asarray(dim)
    shape[I.ndim:] = W
    shape[:I.ndim] -= W - 1
    if np.any(shape < 1):
        raise ValueError('Window size %i is too large for image'%dim)
    P = np.lib.stride_tricks.as_strided(I, shape=shape, strides=strides)
    P = np.reshape(P, [P.shape[0], P.shape[1], dim*dim])
    return P

def getCausalPatches(I, dim):
    """
    Assuming dim is odd, return L-shaped patches that would
    occur in raster order
    """
    P = getPatches(I, dim)
    P = P[:, :, 0:(dim*dim-1)/2]
    return P

def doImageAnalogies(A, Ap, B, NLevels = 3, KSpatial = 5):
    #Make image pyramids
    AL = tuple(pyramid_gaussian(A, NLevels, downscale = 2))
    ApL = tuple(pyramid_gaussian(Ap, NLevels, downscale = 2))
    BL = tuple(pyramid_gaussian(B, NLevels, downscale = 2))
    BpL = []
    print "BL:"
    for i in range(len(BL)):
        print BL[i].shape
        BpL.append(np.zeros(BL[i].shape))
    print "AL:"
    for i in range(len(AL)):
        print AL[i].shape

    #Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        #Step 1: Make features
        APatches = getPatches(rgb2gray(AL[level]), KSpatial)
        ApPatches = getCausalPatches(rgb2gray(ApL[level]), KSpatial)
        X = np.concatenate((APatches, ApPatches), 2)

        B2 = None
        Bp2 = None
        if level < NLevels:
            #Use multiresolution features
            A2 = scipy.misc.imresize(AL[level+1], AL[level].shape)
            Ap2 = scipy.misc.imresize(ApL[level+1], ApL[level].shape)
            A2Patches = getPatches(rgb2gray(A2), KSpatial)
            Ap2Patches = getPatches(rgb2gray(Ap2), KSpatial)
            X = np.concatenate((X, A2Patches, Ap2Patches), 2)
            B2 = scipy.misc.imresize(BL[level+1], BL[level].shape)
            Bp2 = scipy.misc.imresize(BpL[level+1], BpL[level].shape)

        #Compute squared magnitude all feature points
        XSqr = np.sum(X**2, 2)

        #Step 2: Fill in the first few scanLines to prevent the image
        #from getting crap in the beginning
        # if level == NLevels:
        #     BpL[level] = scipy.misc.imresize(ApL[level], BpL[level].shape)
        # else:
        #     BpL[level] = scipy.misc.imresize(BpL[level+1], BpL[level].shape)


        #Step 3: Fill in the pixels in scanline order
        d = (KSpatial-1)/2
        for i in range(d, BpL[level].shape[0]-d):
            for j in range(d, BpL[level].shape[1]-d):
                #Make the feature at this pixel
                #Full patch B
                BPatch = rgb2gray(BL[level][i-d:i+d+1, j-d:j+d+1, :])
                #Causal patch B'
                BpPatch = rgb2gray(BpL[level][i-d:i+d+1, j-d:j+d+1, :]).flatten()
                BpPatch = BpPatch[0:(KSpatial*KSpatial-1)/2]
                F = np.concatenate((BPatch.flatten(), BpPatch.flatten()))

                if level < NLevels:
                    #Use multiresolution features
                    BPatch = rgb2gray(B2[i-d:i+d+1, j-d:j+d+1, :])
                    BpPatch = rgb2gray(Bp2[i-d:i+d+1, j-d:j+d+1, :])
                    F = np.concatenate((F, BPatch.flatten(), BpPatch.flatten()))
                #Find index of most closely matching feature point in A
                DistSqrFn = XSqr - 2*X.dot(F)
                idx = np.unravel_index(np.argmin(DistSqrFn), DistSqrFn.shape)
                BpL[level][i, j, :] = ApL[level][idx[0], idx[1], :]
        writeImage(BpL[level], "%i.png"%level)
    return BpL[0]


if __name__ == '__main__':
    A = readImage("input/blur.A.bmp")
    Ap = readImage("input/blur.Ap.bmp")
    B = readImage("input/blur.B.bmp")
    res = doImageAnalogies(A, Ap, B)

if __name__ == '__main__2':
    idx = np.arange(60)
    [I, J] = np.meshgrid(idx, idx)
    X = (I-30)**2 + (J-30)**2 < 20**2
    P = getPatches(X, 5)
    sio.savemat("P.mat", {"P":P})
    plt.imshow(X)
    plt.show()
