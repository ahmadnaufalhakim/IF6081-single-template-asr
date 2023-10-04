import glob
import numpy as np
import os
from feature_extractor import mfcc_feats

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")

fpaths = [os.path.abspath(fpath) for fpath in glob.glob(os.path.join(DATA_DIR, "hakim", "*.wav"))]

template_mfcc_feats = {
  fpath.split('/')[-1].split('.')[0].lower(): mfcc_feats(fpath) for fpath in fpaths
}

print(template_mfcc_feats)
