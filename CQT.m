function [C] = CQT(X, fs, resol)
  %X: 1D array of audio samples
  %fs: Samplerate
  %resol: Number of frequency bins in each octave
  addpath(genpath('ctwlib_v1.2.0'));
  if nargin == 0
      load('CQTTemp.mat');
  end
  
  %% Constants
  T = length(X(:)); % Time length
  LF = 27.5; % lowest center frequency
  P = 1.0; % calculation range = [-P\sigma,P\sigma] around each center frequency
  alpha = 1.0; % magnitude of obtained CQT is approximately isomorphic to GMM.

  multirate_flag = 0; % if 1, multirate. if 0, timeshift common in all channels

  %% fast approximate CQT with log-normal wavelet
  params = make_wavelet_params(T, fs, resol, LF, P, alpha, multirate_flag);
  C = fastCWT(X(:), params.w, params.L);
  if nargin == 0
      save('CQTTemp.mat', 'C');
  end
end
