import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.io import wavfile
import pyrubberband as pyrb

halfsine = lambda W: np.sin(np.pi*np.arange(W)/float(W))

def STFT(X, W, H, winfunc = None, useLibrosa = True):
    """
    :param X: An Nx1 audio signal
    :param W: A window size
    :param H: A hopSize
    :param winfunc: Handle to a window function
    """
    if useLibrosa:
        import librosa
        return librosa.core.stft(X, n_fft=W, hop_length=H, window = 'blackman')
    Q = W/H
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    if not winfunc:
        #Use half sine by default
        winfunc = halfsine
    win = winfunc(W)
    NWin = int(np.floor((X.size - W)/float(H)) + 1)
    S = np.zeros((W, NWin), dtype = np.complex)
    for i in range(NWin):
        S[:, i] = np.fft.fft(win*X[np.arange(W) + (i-1)*H])
    #Second half of the spectrum is redundant for real signals
    if W%2 == 0:
        #Even Case
        S = S[0:int(W/2)+1, :]
    else:
        #Odd Case
        S = S[0:int((W-1)/2)+1, :]
    return S

def iSTFT(pS, W, H, winfunc = None, useLibrosa = True):
    """
    :param pS: An NBins x NWindows spectrogram
    :param W: A window size
    :param H: A hopSize
    :param winfunc: Handle to a window function
    :returns S: Spectrogram
    """
    if useLibrosa:
        import librosa
        return librosa.core.istft(pS, hop_length = H, window = 'blackman')
    #First put back the entire redundant STFT
    S = np.array(pS, dtype = np.complex)
    if W%2 == 0:
        #Even Case
        S = np.concatenate((S, np.flipud(np.conj(S[1:-1, :]))), 0)
    else:
        #Odd Case
        S = np.concatenate((S, np.flipud(np.conj(S[1::, :]))), 0)
    
    #Figure out how long the reconstructed signal actually is
    N = W + H*(S.shape[1] - 1)
    X = np.zeros(N, dtype = np.complex)
    
    #Setup the window
    Q = W/H;
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    if not winfunc:
        #Use half sine by default
        winfunc = halfsine
    win = winfunc(W)
    win = win/(Q/2.0)

    #Do overlap/add synthesis
    for i in range(S.shape[1]):
        X[i*H:i*H+W] += win*np.fft.ifft(S[:, i])
    return X

def pitchShiftSTFT(S, Fs, shift):
    """
    Use image interpolation to do spectogram shifting in the spectral domain
    :param S: An NFreqsxNWindows spectrogram
    :param Fs: Sample rate
    :param shift: Number of halfsteps by which to shift
    :returns: An NFreqsxNWindows shifted spectrogram
    """
    M = S.shape[0]
    N = S.shape[1]
    bins0 = np.arange(M)
    freqs0 = Fs*bins0/float(M)
    freqs1 = freqs0*2.0**(-shift/12.0)
    bins1 = (freqs1/Fs)*M
    wins = np.arange(N)
    print("freqs0.shape = ", freqs0.shape)
    print("wins.shape = ", wins.shape)
    print("S.shape = ", S.shape)
    f = scipy.interpolate.interp2d(wins, freqs0, S, kind = 'linear')
    return f(wins, freqs1)

def getMelFilterbank(Fs, winSize, noctaves = 7, binsperoctave = 24):
    """
    Return a mel-spaced triangular filterbank
    :param Fs: Audio sample rate
    :param winSize: Window size of associated STFT
    :param noctaves: Number of octaves to cover, starting at A0
    :param binsperoctave: Number of bins per octave
    :returns melfbank: An NBands x NSpectrumSamples matrix
        with each filter per row
    """
    fmin = 55.0
    fmax = fmin*(2.0**noctaves)
    NBands = noctaves*binsperoctave
    NSpectrumSamples = int(winSize/2)+1

    melbounds = np.array([fmin, fmax])
    melbounds = 1125*np.log(1 + melbounds/700.0)
    mel = np.linspace(melbounds[0], melbounds[1], NBands+2)
    binfreqs = 700*(np.exp(mel/1125.0) - 1)
    binbins = np.floor(((winSize-1)/float(Fs))*binfreqs) #Floor to the nearest bin
    binbins = np.array(binbins, dtype=np.int64)

    #Step 2: Create mel triangular filterbank
    melfbank = np.zeros((NBands, NSpectrumSamples))
    for i in range(1, NBands+1):
        thisbin = binbins[i]
        lbin = binbins[i-1]
        rbin = thisbin + (thisbin - lbin)
        rbin = binbins[i+1]
        melfbank[i-1, lbin:thisbin+1] = np.linspace(0, 1, 1 + (thisbin - lbin))
        melfbank[i-1, thisbin:rbin+1] = np.linspace(1, 0, 1 + (rbin - thisbin))
    melfbank = melfbank/np.sum(melfbank, 1)[:, None]
    return melfbank

def warpSTFTMel(S, Fs, winSize):
    M = getMelFilterbank(Fs, winSize)
    plt.show()
    return M.dot(S)

def unwarpSTFTMel(X, Fs, winSize):
    M = getMelFilterbank(Fs, winSize)
    MEnergy = np.sum(M, 0)
    idxpass = (MEnergy == 0)
    MEnergy[idxpass] = 1
    M = M/MEnergy[None, :]
    return (M.T).dot(X)

def getPitchShiftedSpecs(X, Fs, W, H, shiftrange = 6):
    """
    Concatenate a bunch of pitch shifted versions of the spectrograms
    of a sound, using the rubberband library
    :param X: A mono audio array
    :param Fs: Sample rate
    :param W: Window size
    :param H: Hop size
    :param shiftrange: The number of halfsteps below and above which \
        to shift the sound
    :returns SRet: The concatenate spectrogram
    """
    SRet = np.array([])
    for shift in range(-shiftrange, shiftrange+1):
        print("Computing STFT pitch shift %i"%shift)
        if shift == 0:
            Y = np.array(X)
        else:
            Y = pyrb.pitch_shift(X, Fs, shift)
        S = STFT(Y, W, H)
        if SRet.size == 0:
            SRet = S
        else:
            SRet = np.concatenate((SRet, S), 1)
    return SRet

def griffinLimInverse(S, W, H, NIters = 10, winfunc = None):
    """
    Do Griffin Lim phase retrieval
    :param S: An NFreqsxNWindows spectrogram
    :param W: Window size used in STFT
    :param H: Hop length used in STFT
    :param NIters: Number of iterations to go through (default 10)
    :winfunc: A handle to a window function (None by default, use halfsine)
    :returns: An Nx1 real signal corresponding to phase retrieval
    """
    eps = 2.2204e-16
    if not winfunc:
        winfunc = halfsine
    A = np.array(S, dtype = np.complex)
    for i in range(NIters):
        print("Iteration %i of %i"%(i+1, NIters))
        A = STFT(iSTFT(A, W, H, winfunc), W, H, winfunc)
        Norm = np.sqrt(A*np.conj(A))
        Norm[Norm < eps] = 1
        A = S*(A/Norm)
    X = iSTFT(A, W, H, winfunc)
    return np.real(X)

def griffinLimCQTInverse(C, Fs, H, bins_per_octave, NIters = 10):
    """
    Do Griffin Lim inverse using CQT instead of STFT
    :param C: An NBinsxNWindows CQT
    :param Fs: Sample rate
    :param H: Hop length used in C
    :param bins_per_octave: Number of bins per octave
    :param NIters: Number of iterations to go through (default 10)
    :returns: An Nx1 real signal corresponding to phase retrieval
    """
    import librosa2
    eps = 2.2204e-16
    A = np.array(C, dtype = np.complex)
    for i in range(NIters):
        print("Iteration %i of %i"%(i+1, NIters))
        x = librosa2.icqt(C=A, sr=Fs, hop_length=H, \
                     bins_per_octave=bins_per_octave)
        A = librosa2.cqt(y=x, sr=Fs, hop_length=H, n_bins=A.shape[0],\
               bins_per_octave=bins_per_octave)
        Norm = np.sqrt(A*np.conj(A))
        Norm[Norm < eps] = 1
        A = C*(A/Norm)
    X = librosa2.icqt(C=A, sr=Fs, hop_length=H, \
                     bins_per_octave=bins_per_octave)
    return np.real(X)

def testPitchShift(X, Fs, W, H, shift, filename):
    W = 2048
    H = 128
    S = np.abs(STFT(X, W, H))
    S = pitchShiftSTFT(S, Fs, shift)
    X2 = griffinLimInverse(S, W, H, 20)
    wavfile.write(filename, Fs, X2)

def testPitchShiftCQT(X, Fs, H, bins_per_octave, shift, filename, noctaves):
    from scipy.io import wavfile
    import librosa2
    import librosa2.display
    C = librosa2.cqt(y=X, sr=Fs, hop_length=H, n_bins=noctaves*bins_per_octave,\
               bins_per_octave=bins_per_octave)
    print("C.shape = ", C.shape)
    C2 = np.zeros(C.shape, dtype = np.complex)
    k = int(bins_per_octave/12)
    if abs(shift) > 0:
        C2[k*shift::, :] = C[0:-k*shift, :]
    else:
        C2 = C
    plt.subplot(211)
    librosa2.display.specshow(librosa.amplitude_to_db(C, ref=np.max),\
                sr=Fs, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    plt.subplot(212)
    librosa2.display.specshow(librosa.amplitude_to_db(C2, ref=np.max),\
                sr=Fs, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    plt.title("After Shift By %i"%shift)
    plt.show()
    x = librosa2.icqt(C=C2, sr=Fs, hop_length=H, \
                    bins_per_octave=bins_per_octave)
    wavfile.write("complexicqt_%s"%filename, Fs, x)

    C2 = np.abs(C2)
    y_hat = griffinLimCQTInverse(C2, Fs, H, bins_per_octave)
    y_hat = y_hat/np.max(np.abs(y_hat))
    wavfile.write(filename, Fs, y_hat)

if __name__ == '__main__':
    import librosa
    import pyrubberband as pyrb
    X, Fs = librosa.load("music/Beatles_LetItBe.mp3")
    shift = 2
    noctaves = 7
    y = pyrb.pitch_shift(X, Fs, shift)
    wavfile.write("rubberbandshift%i.wav"%shift, Fs, y)
    testPitchShift(X, Fs, 2048, 128, shift, "gfshift%i_stft.wav"%shift)
    """
    testPitchShiftCQT(y, Fs, 128, 36, shift, \
            "gfshift%i_cqt_%ioctaves.wav"%(shift, noctaves), noctaves)
    """