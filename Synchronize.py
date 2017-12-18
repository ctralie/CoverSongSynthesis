"""
Purpose: To 
"""
import numpy as np
import os
import glob
import scipy.io as sio
import scipy.misc
import time
import matplotlib.pyplot as plt
from GeometricCoverSongs.CSMSSMTools import *
from GeometricCoverSongs.BlockWindowFeatures import *
from GeometricCoverSongs.pyMIRBasic.AudioIO import *
from GeometricCoverSongs.pyMIRBasic.Onsets import *
import json
import pyrubberband as pyrb
import subprocess

def getGreedyPerm(D):
    """
    Purpose: Naive O(N^2) algorithm to do the greedy permutation
    param: D (NxN distance matrix for points)
    return: (permutation (N-length array of indices),
            lambdas (N-length array of insertion radii))
    """
    N = D.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def syncBlocks(path, CSM, beats1, beats2, Fs, hopSize, XAudio1, XAudio2, BeatsPerBlock, fileprefix = ""):
    """
    :param path: Px2 array representing a partial warping path to align two songs
    :param CSM: The cross similarity matrix between two songs
    :param beats1: An array of beat onsets for song 1 in increments of hopSize
    :param beats2: An array of beat onsets for song 2 in increments of hopSize
    :param XAudio1: The raw audio samples for song 1
    :param XAudio2: The raw audio samples for song 2
    :param BeatsPerBlock: The number of beats per block for each pixel in the CSM
    :param fileprefix: Prefix of each stretched block to save.  By default, blank,\
        so no debugging info saved
    :returns (XFinal: An NSamples x 2 array with the first song along the first column\
                and the second synchronized song along the second column,\
              beatsFinal: An array of the locations in samples of the beat onsets in XFinal \
              scoresFinal: An array of matching scores for each beat)
    """
    XFinal = np.array([[0, 0]])
    beatsFinal = [] #The final beat locations based on hop size
    scoresFinal = []
    for i in range(path.shape[0]):
        [j, k] = [path[i, 0], path[i, 1]]
        if j >= CSM.shape[0] or k >= CSM.shape[1]:
            break
        scoresFinal.append(CSM[j, k])
        t1 = beats1[j]*hopSize
        t2 = beats1[j+BeatsPerBlock]*hopSize
        s1 = beats2[k]*hopSize
        s2 = beats2[k+BeatsPerBlock]*hopSize
        x1 = XAudio1[t1:t2]
        x2 = XAudio2[s1:s2]
        #Figure out the time factor by which to stretch x2 so it aligns
        #with x1
        fac = float(len(x1))/len(x2)
        print("fac = ", fac)
        x2 = pyrb.time_stretch(x2, Fs, 1.0/fac)
        print("len(x1) = %i, len(x2) = %i"%(len(x1), len(x2)))
        N = min(len(x1), len(x2))
        x1 = x1[0:N]
        x2 = x2[0:N]
        X = np.zeros((N, 2))
        X[:, 0] = x1
        X[:, 1] = x2
        if len(fileprefix) > 0:
            filename = "%s_%i.mp3"%(fileprefix, i)
            sio.wavfile.write("temp.wav", Fs, X)
            subprocess.call(["avconv", "-i", "temp.wav", filename])
        beat1 = beats1[j+1]*hopSize-t1
        beatsFinal.append(XFinal.shape[0])
        XFinal = np.concatenate((XFinal, X[0:beat1, :]))
    return (XFinal, beatsFinal, scoresFinal)

def expandBeats(beats, bSub):
    import scipy.interpolate as interp
    idx = np.arange(beats.size)
    idxx = (np.arange(bSub*beats.size)/float(bSub))[0:-bSub+1]
    y = interp.spline(idx, beats, idxx)
    y = np.round(y)
    return np.array(y, dtype = np.int64)

def synchronize(filename1, filename2, hopSize, TempoBiases, bSub, FeatureParams, CSMTypes,\
                 Kappa, fileprefix = "", doPlot = False, outputSnippets = True, doNegative = False):
    """
    :param filename1: First song path
    :param filename2: Second song path
    :param hopSize: Hop size (in samples) to be used between feature windows
    :param TempoBiases: The different tempo levels to be tried for beat tracking
    :param bSub: The factor by which to subdivide the beats
    :param FeatureParams: Dictionary of feature parameters
    :param CSMTypes: Dictionary of CSM types for different features
    :param Kappa: Nearest neighbor fraction for making binary CSM
    :param fileprefix: File prefix for debugging plots and intermediate audio files
    :param doPlot: Whether to plot alignment
    :param outputSnippets: Whether to output aligned audio snippets block by block
    :param doNegative: Whether to sample negative matches
    """
    print "Loading %s..."%filename1
    (XAudio1, Fs) = getAudioLibrosa(filename1)
    print "Loading %s..."%filename2
    (XAudio2, Fs) = getAudioLibrosa(filename2)

    maxScore = 0.0
    maxRes = {}

    for TempoBias1 in TempoBiases:
        for TempoBias2 in TempoBiases:
            print "Doing TempoBias1 = %i, TempoBias2 = %i..."%(TempoBias1, TempoBias2)
            (tempo, beats1) = getBeats(XAudio1, Fs, TempoBias1, hopSize, filename1)
            beats1 = expandBeats(beats1, bSub)
            (Features1, O1) = getBlockWindowFeatures((XAudio1, Fs, tempo, beats1, hopSize, FeatureParams))
            (tempo, beats2) = getBeats(XAudio2, Fs, TempoBias2, hopSize, filename2)
            beats2 = expandBeats(beats2, bSub)
            (Features2, O2) = getBlockWindowFeatures((XAudio2, Fs, tempo, beats2, hopSize, FeatureParams))
            print "Doing similarity fusion"
            K = 20
            NIters = 3
            res = getCSMSmithWatermanScoresEarlyFusionFull(Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes, doPlot = True, conservative = False)
            print "score = ", res['score']
            if res['score'] > maxScore:
                print "New maximum score!"
                maxScore = res['score']
                maxRes = res
                res['beats1'] = beats1
                res['beats2'] = beats2
                res['TempoBias1'] = TempoBias1
                res['TempoBias2'] = TempoBias2
    res = maxRes
    print("TempoBias1 = %i, TempoBias2 = %i"%(res['TempoBias1'], res['TempoBias2']))
    beats1 = res['beats1']
    beats2 = res['beats2']
    CSM = res['CSM']
    CSM = CSM/np.max(CSM) #Normalize so highest score is 1
    path = np.array(res['path'])

    if doPlot:
        plt.clf()
        plt.figure(figsize=(20, 8))
        plt.subplot(121)
        plt.imshow(CSM, cmap = 'afmhot')
        plt.hold(True)
        plt.plot(path[:, 1], path[:, 0], '.')
        plt.subplot(122)
        plt.plot(path[:, 0], path[:, 1])
        plt.savefig("%sBlocksAligned.svg"%fileprefix, bbox_inches = 'tight')

    #Now extract signal snippets that are in correspondence, beat by beat
    BeatsPerBlock = FeatureParams['MFCCBeatsPerBlock']
    path = np.flipud(path)
    (XFinal, beatsFinal, scoresFinal) = syncBlocks(path, CSM, beats1, beats2, Fs, hopSize, XAudio1, XAudio2, BeatsPerBlock, fileprefix = "")
    #Write out true positives synced
    if len(fileprefix) > 0:
        sio.wavfile.write("temp.wav", Fs, XFinal)
        subprocess.call(["avconv", "-i", "temp.wav", "%sTrue.mp3"%fileprefix])
    #Write out true positives beat times and scores
    [beatsFinal, scoresFinal] = [np.array(beatsFinal), np.array(scoresFinal)]
    if len(fileprefix) > 0:
        sio.savemat("%sTrue.mat"%fileprefix, {"beats":beatsFinal, "scores":scoresFinal, "BeatsPerBlock":BeatsPerBlock, "hopSize":hopSize})

    #Now save negative examples (same number as positive blocks)
    if doNegative:
        NBlocks = path.shape[0]
        x = CSM.flatten()
        idx = np.argsort(x)
        idx = idx[0:5*CSM.shape[0]]
        idxy = np.unravel_index(idx, CSM.shape)
        idx = np.zeros((idx.size, 2), dtype = np.int64)
        idx[:, 0] = idxy[0]
        idx[:, 1] = idxy[1]
        D = getCSM(idx, idx)
        #Do furthest point sampling on negative locations
        (perm, lambdas) = getGreedyPerm(D)
        path = idx[perm[0:NBlocks], :]
        if doPlot:
            plt.clf()
            plt.imshow(CSM, interpolation = 'nearest', cmap = 'afmhot')
            plt.hold(True)
            plt.plot(path[:, 1], path[:, 0], '.')
            plt.savefig("%sBlocksMisaligned.svg"%fileprefix, bbox_inches = 'tight')
        #Output negative example audio synced
        (XFinal, beatsFinal, scoresFinal) = syncBlocks(path, CSM, beats1, beats2, Fs, hopSize, XAudio1, XAudio2, BeatsPerBlock, fileprefix = "%sFalse"%fileprefix)
        sio.savemat("%sFalse.mat"%fileprefix, {"scores":scoresFinal, "BeatsPerBlock":BeatsPerBlock, "hopSize":hopSize})
    return {'X':XFinal, 'Fs':Fs, 'beatsFinal':beatsFinal, 'scoresFinal':scoresFinal}

if __name__ == '__main__':
    Kappa = 0.1
    hopSize = 512
    TempoBiases = [0]
    
    """
    filename1 = "DespacitoOrig.mp3"
    filename2 = "DespacitoMetal.mp3"
    fileprefix = "Despacito" #Save a JSON file with this prefix
    artist1 = "Luis Fonsi ft. Daddy Yankee"
    artist2 = "Leo Moracchioli"
    songName = "Despacito"
    """
    
    """
    filename1 = "WakaNoHands.webm"
    filename2 = "DannyVolaNoHands.m4a"
    artist1 = "Waka Flocka Flame"
    artist2 = "Danny Vola"
    fileprefix = "nohands"
    songName = "No Hands"
    """

    """
    filename1 = "LaFolia1.mp3"
    filename2 = "LaFolia2.mp3"
    artist1 = "Vivaldi"
    artist2 = "Vivaldi"
    fileprefix = "LaFolia"
    songName = "La Folia"
    """

    filename1 = "music/SmoothCriminalMJ.mp3"
    filename2 = "music/SmoothCriminalAAF.mp3"
    artist1 = "Michael Jackson"
    artist2 = "Alien Ant Farm"
    fileprefix = "smoothcriminal"
    songName = "Smooth Criminal"
    TempoBiases = [180]


    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    res = synchronize(filename1, filename2, hopSize, TempoBiases, 2, FeatureParams, CSMTypes, Kappa)
    sio.wavfile.write("temp.wav", res['Fs'], res['X'])
    subprocess.call(["avconv", "-i", "temp.wav", "%sTrue.mp3"%fileprefix])