import numpy as np
from python_speech_features import mfcc, delta
import scipy.io.wavfile as wav

WINLEN = 0.02
NFFT = 1024

def extract_mfcc_feats(fpath:str) :
  (rate,sig) = wav.read(fpath)

  mfcc_feats = mfcc(sig, rate, winlen=WINLEN, nfft=NFFT)
  delta_mfcc_feats = delta(mfcc_feats, 1)
  delta_delta_mfcc_feats = delta(delta_mfcc_feats, 1)
  num_frames = mfcc_feats.shape[0]

  all_mfcc_feats = np.array([mfcc_feats, delta_mfcc_feats, delta_delta_mfcc_feats]) \
    .transpose(1, 0, 2) \
    .reshape(num_frames, -1)
  return all_mfcc_feats
