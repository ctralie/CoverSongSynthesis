import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

def getPrefix():
    s = ''
    return s

def initMatlabEngine():
    prefix = getPrefix()
    import matlab.engine
    beforePath = os.path.realpath(".")
    if len(prefix) > 0:
        os.chdir('%s'%prefix)
    eng = matlab.engine.start_matlab()
    os.chdir(beforePath)
    return eng

def getCQTNakamuraMatlab(eng, X, fs, resol=24, memCall = False):
    """
    Wrap around Nakamura's Matalb code to compute the CQT
    :param eng: Matlab engine handle
    :param X: A 1D array of audio samples
    :param Fs: Sample rate
    :param resol: Number of CQT bins per octave
    :param memCall: Whether to keep arrays/params in memory or to save them to the \
        hard drive in a temporary array (latter is faster for some reason)
    """
    if memCall:
        import matlab.engine
        print("Converting audio file to Matlab array...")
        XParam = matlab.double(X.tolist())
        print("Computing CQT...")
        res = eng.CQT(XParam, float(fs), float(resol))
        return np.array(res)
    else:
        sio.savemat("CQTTemp.mat", {"X":X, "fs":float(fs), "resol":float(resol)})
        eng.CQT()
        return sio.loadmat("CQTTemp.mat")["C"]

def getiCQTNakamuraMatlab(eng, C, T, fs, resol, memCall = False):
    """
    Wrap around Nakamura's Matalb code to compute the iCQT
    :param eng: Matlab engine handle
    :param C: CQT array
    :param T: Length of signal
    :param Fs: Sample rate
    :param resol: Number of CQT bins per octave
    :param memCall: Whether to keep arrays/params in memory or to save them to the \
        hard drive in a temporary array (latter is faster for some reason)
    """
    if memCall:
        import matlab.engine
        print("Converting iCQT parameters to Matlab array..")
        CReal = matlab.double((np.real(C)).tolist())
        CImag = matlab.double((np.imag(C)).tolist())
        print("Computing iCQT...")
        res = eng.iCQT(CReal, CImag, float(T), float(fs), float(resol))
        return np.array(res)
    else:
        sio.savemat("CQTTemp.mat", {"C":C, "T":float(T), "fs":float(fs), \
            "resol":float(resol)})
        eng.iCQT()
        return sio.loadmat("CQTTemp.mat")['rec']

def getiCQTGriffinLimNakamuraMatlab(eng, C, T, fs, resol, NIters = 20, \
        randPhase = False, memCall = False):
    """
    Wrap around Nakamura's Matalb code to compute Griffin Lim iCQT
    :param eng: Matlab engine handle
    :param C: CQT array
    :param T: Length of signal
    :param Fs: Sample rate
    :param resol: Number of CQT bins per octave
    :param randPhase: If true, multiply C by a random phase
    :param memCall: Whether to keep arrays/params in memory or to save them to the \
        hard drive in a temporary array (latter is faster for some reason)
    """
    if randPhase:
        C = np.exp(np.complex(0, 1)*np.random.rand(C.shape[0], C.shape[1]))*C
    if memCall:
        import matlab.engine
        print("Converting iCQT parameters to Matlab array..")
        CReal = matlab.double((np.real(C)).tolist())
        CImag = matlab.double((np.imag(C)).tolist())
        print("Computing iCQT...")
        res = eng.iCQTGriffinLim(CReal, CImag, float(T), float(fs), float(resol), float(NIters))
        print(res)
        return (np.array(res['rec_sig']), np.array(res['rec_spec']))
    else:
        sio.savemat("CQTTemp.mat", {"C":C, "T":float(T), "fs":float(fs), \
            "resol":float(resol), "NIters":NIters})
        eng.iCQTGriffinLim()
        res = sio.loadmat("CQTTemp.mat")
        return (res["rec_sig"], res["rec_spec"])

def getNSGT(X, Fs, resol=24):
    """
    Perform a Nonstationary Gabor Transform implementation of CQT
    :param X: A 1D array of audio samples
    :param Fs: Sample rate
    :param resol: Number of CQT bins per octave
    """
    from nsgt import NSGT,OctScale
    scl = OctScale(50, Fs, resol)
    nsgt = NSGT(scl, Fs, len(X), matrixform=True)
    C = nsgt.forward(X)
    return np.array(C)

def getiNSGT(C, L, Fs, resol=24):
    """
    Perform an inverse Nonstationary Gabor Transform
    :param C: An NBinsxNFrames CQT array
    :param L: Number of samples in audio file
    :param Fs: Sample rate
    :param resol: Number of CQT bins per octave
    """
    from nsgt import NSGT,OctScale
    scl = OctScale(50, Fs, resol)
    nsgt = NSGT(scl, Fs, L, matrixform=True)
    return nsgt.backward(C)

def getiNSGTGriffinLim(C, L, Fs, resol=24, randPhase = False, NIters = 20):
    from nsgt import NSGT,OctScale
    scl = OctScale(50, Fs, resol)
    nsgt = NSGT(scl, Fs, L, matrixform=True)
    eps = 2.2204e-16
    if randPhase:
        C = np.exp(np.complex(0, 1)*np.random.rand(C.shape[0], C.shape[1]))*C
    A = np.array(C, dtype = np.complex)
    for i in range(NIters):
        print("iNSGT Griffin Lim Iteration %i of %i"%(i+1, NIters))
        A = nsgt.forward(nsgt.backward(C))
        Norm = np.sqrt(A*np.conj(A))
        Norm[Norm < eps] = 1
        A = np.abs(C)*(A/Norm)
    X = nsgt.backward(A)
    return np.real(X)


def testNakamura(X, Fs):
    """
    Test Nakamura's technique on phase retrieval 
    with and without pitch shifting
    """
    from scipy.io import wavfile
    eng = initMatlabEngine()
    C = getCQTNakamuraMatlab(eng, X, Fs)
    plt.imshow(np.abs(C), cmap = 'afmhot', aspect = 'auto')
    plt.show()
    #Test regular CQT
    Y = getiCQTNakamuraMatlab(eng, C, X.size, Fs, 24)
    wavfile.write("reconNak.wav", Fs, Y)
    #Test phase retrieval CQT with pitch shifting
    C = np.abs(C)
    C2 = np.zeros(C.shape)
    steps = 3
    C2[0:-steps*2, :] = C[steps*2::, :]
    (Z, CRec) = getiCQTGriffinLimNakamuraMatlab(eng, C2, X.size, Fs, 24, \
        NIters=100, randPhase = True)
    wavfile.write("reconShiftNak.wav", Fs, Z)

def testNSGT(X, Fs, NIters = 100):
    """
    Test Thomas Grill's NSGT technique on phase retrieval 
    with and without pitch shifting
    """
    C = getNSGT(X, Fs)
    C = np.abs(C)
    #Test regular inversion
    Y = getiNSGTGriffinLim(C, len(X), Fs, randPhase=True, NIters = NIters)
    wavfile.write("reconNSGT.wav", Fs, Y)
    #Test with pitch shifting
    C2 = np.zeros(C.shape)
    steps = 3
    C2[0:-steps*2, :] = C[steps*2::, :]
    Z = getiNSGTGriffinLim(C2, len(X), Fs, randPhase=True, NIters = NIters)
    wavfile.write("reconShiftNSGT.wav", Fs, Z)

if __name__ == '__main__':
    from scipy.io import wavfile
    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    A = X[:, 0]/(2.0**15)
    A = A[0:Fs*30]
    testNSGT(A, Fs)
    testNakamura(A, Fs)