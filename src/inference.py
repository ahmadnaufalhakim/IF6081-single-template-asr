from dtw import dtw
from feature_extractor import extract_mfcc_feats
import numbers

def infer(words_to_template_mfcc_feats:dict, input_fpath:str, mode:str=None) :
  input_mfcc_feats = extract_mfcc_feats(input_fpath)
  if mode is None or mode == "euclidean" :
    p = 2 # Use Euclidean distance as default
  elif mode == "manhattan" :
    p = 1
  elif isinstance(mode, numbers.Number) :
    p = mode
  words = words_to_template_mfcc_feats.keys()
  best_word, best_cost, best_path = '', float("inf"), []
  for word in words :
    cost, path = dtw(
      x=input_mfcc_feats,
      y=words_to_template_mfcc_feats[word],
      dist=p
    )
    if cost < best_cost :
      best_word, best_cost, best_path = word, cost, path
  return best_word, best_cost, best_path
