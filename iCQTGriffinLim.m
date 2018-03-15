function [ rec_sig, rec_spec ] = iCQTGriffinLim( CReal, CImag, T, fs, resol, NIters )
  addpath(genpath('ctwlib_v1.2.0'));
  if nargin == 0
      load('CQTTemp.mat');
  else
      C = CReal + 1i*CImag;
  end
  
  
  %% Constants
  LF = 27.5; % lowest center frequency
  P = 1.0; % calculation range = [-P\sigma,P\sigma] around each center frequency
  alpha = 1.0; % magnitude of obtained CQT is approximately isomorphic to GMM.
  multirate_flag = 0; % if 1, multirate. if 0, timeshift common in all channels

  %% fast approximate CQT with log-normal wavelet
  params = make_wavelet_params(T, fs, resol, LF, P, alpha, multirate_flag);

  %% Signal reconstruction from magnitude CQT spectrogram
  [rec_sig,rec_spec] = phaserec_fastCWT(C,params,NIters, 0);
  if nargin == 0
      save('CQTTemp.mat', 'rec_sig', 'rec_spec');
  end
end

