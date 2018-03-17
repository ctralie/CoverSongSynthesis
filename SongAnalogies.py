import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from scipy.io import wavfile
import scipy.io as sio
import pyrubberband as pyrb
from Synchronize import *
from GeometricCoverSongs.CSMSSMTools import *
from GeometricCoverSongs.pyMIRBasic.Chroma import *
from GeometricCoverSongs.pyMIRBasic.MFCC import *

def getNormedFeatures(S, winSize, hopSize):
    """
    
    """
    Norms = np.sqrt(np.sum(X**2, 1))
    Norms[Norms == 0] = 1
    return X/Norms[:, None]

def doSpectrogramAnalogiesSimple(SA, SAp, SB, Kappa):
    #Compute normalized spectrogram similarities for each window
    SBp = np.zeros_like(SB)
    MagSA = np.sqrt(np.abs(np.sum(SA*np.conj(SA), 0)))
    MagSA[MagSA == 0] = 1
    MagSB = np.sqrt(np.abs(np.sum(SB*np.conj(SB), 0)))
    MagSB[MagSB == 0] = 1
    SA = SA/MagSA[None, :]
    SB = SB/MagSB[None, :]
    SAM = np.abs(SA)
    for k in range(SB.shape[1]):
        x = np.abs(SB[:, k].flatten())
        D = getCSM(x[None, :], SAM.T).flatten()
        idx = np.argmin(D)
        dist = D[idx]
        if k > 0:
            if D[idx-1] < (1+Kappa)*dist:
                idx = idx-1
                dist = D[idx]
        #Scale by power ratio
        SBp[:, k] = SAp[:, idx]*MagSB[k]/MagSA[idx]
    return SBp

def getMFCCsFromSpec(S, Fs, NBands = 40, fmax = 8000, NMFCC = 20, lifterexp = 0.6):
    winSize = (S.shape[0]-1)*2
    M = librosa.filters.mel(Fs, winSize, n_mels = NBands, fmax = fmax)
    X = M.dot(np.abs(S))
    X = librosa.core.logamplitude(X)
    X = np.dot(librosa.filters.dct(NMFCC, X.shape[0]), X) #Make MFCC
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    X = coeffs[:, None]*X
    X = np.array(X, dtype = np.float32)
    return X


def doSpectrogramAnalogies(SA, SAp, SB, T, Fs, useMFCCs = True):
    N = SA.shape[1]
    M = N-T+1
    if useMFCCs:
        MA = getMFCCsFromSpec(SA, Fs)
        MAp = getMFCCsFromSpec(SAp, Fs)
        MB = getMFCCsFromSpec(SB, Fs)
    else:
        (MA, MAp, MB) = (SA, SAp, SB)
    #Step 1: Make sliding windows for A and Ap
    SWs = []
    for i in range(M):
        S = np.concatenate((MA[:, i:i+T], MAp[:, i:i+T-1]), 1)
        S = S.flatten()
        S = S/np.sqrt(np.sum(S**2))
        SWs.append(S)
    SWs = np.array(SWs)
    #Step 2: Step through and fill in windows for SB
    N = SB.shape[1]
    M = N-T+1
    SBp = np.zeros_like(SB)
    MBp = np.zeros_like(MB)
    #Fill in some stuff at the beginning
    SBp[:, 0:T-1] = SAp[:, 0:T-1]
    MBp[:, 0:T-1] = MAp[:, 0:T-1]
    idxs = []
    for i in range(T-1, M):
        S = np.concatenate((MB[:, i-T+1:i+1], MBp[:, i-T+1:i]), 1)
        S = S.flatten()
        S = S/np.sqrt(np.sum(S**2))
        D = getCSM(S[None, :], SWs)
        idx = np.argmin(D.flatten())
        idxs.append(idx)
        s = SAp[:, idx+T-1]
        SBp[:, i] = s
        MBp[:, i] = getMFCCsFromSpec(s[:, None], Fs).flatten()
    idxs = np.array(idxs)
    return SBp

def makeAnalogy(X, Fs, beatsA, filename_b, hopSize, winSize, ws, TempoBias, MFCCWeight = 1.0, HPCPWeight = 1.0):
    """
    Make a cover song analogy; given audio for (A, A'), and B, \
        make B'
    :param X: Audio waveform for A and A'; A is in first column, A' in second
    :param Fs: Sample rate of all audio files
    :param beatsA: Beat onsets (in samples)
    :param hopSize: Feature hop size
    :param winSize: Window size for MFCCs and HPCPs
    :param ws: Window weights of all features
    :param TempoBias: Tempo bias for beat tracking of song B
    """
    #Step 1: Load in new example from artist 1 (B song)
    print("Loading new example...")
    XA = X[:, 0]
    XAp = X[:, 1]
    XB, Fs2 = librosa.load(filename_b)
    XB = librosa.core.to_mono(XB)

    #Step 2: Use rubberband library to change tempo of B so that
    #it's in line with tempo of song A
    tempoB, beatsB = librosa.beat.beat_track(XB, Fs2, start_bpm = TempoBias, hop_length = hopSize)
    tempoA = 60.0/(np.mean(beatsA[1::] - beatsA[0:-1])/float(Fs))
    print("tempoA = %g, tempoB = %g"%(tempoA, tempoB))
    ratio = float(tempoA)/tempoB
    print("Shifting by ratio: %g"%ratio)
    XB = pyrb.time_stretch(XB, Fs2, ratio)

if __name__ == '__main__':
    #Sync parameters
    filename1 = "music/SmoothCriminalMJ.mp3"
    filename2 = "music/SmoothCriminalAAF.mp3"
    artist1 = "Michael Jackson"
    artist2 = "Alien Ant Farm"
    jointfileprefix = "SmoothCriminalSync"
    filename_b = "music/MJBad.mp3"
    TempoBiases = [180]
    bSub = 1
    Kappa = 0.1
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Analogy parameters
    ws = np.ones(1)
    hopSize = 512
    #TODO: Tweak window size
    #(tradeoff for being localized in time but also not sensitive to window position)
    winSize = hopSize*4

    #Step 1: Load in and synchronize example pair
    print("Loading and synchronizing example pair...")
    if os.path.exists("%s.mat"%jointfileprefix):
        res = sio.loadmat("%s.mat"%jointfileprefix)
        [X, Fs, beats] = [res['X'], res['Fs'], res['beatsFinal']]
    else:
        res = synchronize(filename1, filename2, hopSize, TempoBiases, bSub, FeatureParams, CSMTypes, Kappa)
        [X, Fs, beats] = [res['X'], res['Fs'], res['beatsFinal']]
        sio.savemat("%s.mat"%jointfileprefix, res)
        sio.wavfile.write("%s.wav"%jointfileprefix, Fs, X)
    beats = beats.flatten()
    #X = np.array(X, dtype = np.single)

    makeAnalogy(X, Fs, beats, filename_b, hopSize, winSize, ws, TempoBiases[0])