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



if __name__ == '__main__':
    from scipy.io import wavfile
    eng = initMatlabEngine()
    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    A = X[:, 0]/(2.0**15)
    A = A[0:Fs*30]
    C = getCQTNakamuraMatlab(eng, A, Fs)
    plt.imshow(np.abs(C), cmap = 'afmhot', aspect = 'auto')
    plt.show()
    #Test regular CQT
    Y = getiCQTNakamuraMatlab(eng, C, A.size, Fs, 24)
    wavfile.write("recon.wav", Fs, Y)
    #Test phase retrieval CQT with pitch shifting
    C = np.abs(C)
    C2 = np.zeros(C.shape)
    steps = 3
    C2[0:-steps*2, :] = C[steps*2::, :]
    (Z, CRec) = getiCQTGriffinLimNakamuraMatlab(eng, C2, A.size, Fs, 24, \
        NIters=100, randPhase = True)
    wavfile.write("reconphase.wav", Fs, Z)