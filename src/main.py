import glob
import os
from sklearn.metrics import accuracy_score

from feature_extractor import extract_mfcc_feats
from inference import infer

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")

fpaths = [os.path.abspath(fpath) for fpath in glob.glob(os.path.join(DATA_DIR, "nio", "*.wav"))]

words_to_template_mfcc_feats = {
  fpath.split('/')[-1].split('.')[0].lower(): extract_mfcc_feats(fpath) for fpath in fpaths
}

input_fpaths = [os.path.abspath(fpath) for fpath in glob.glob(os.path.join(DATA_DIR, "hakim", "*.wav"))]
input_fpaths_to_label_and_mfcc_feats = {
  input_fpath: (input_fpath.split('/')[-1].split('.')[0].lower(), extract_mfcc_feats(input_fpath)) for input_fpath in input_fpaths
}

labels, predictions = [], []
for input_fpath in input_fpaths_to_label_and_mfcc_feats :
  label = input_fpaths_to_label_and_mfcc_feats[input_fpath][0]
  predicted_word, cost, path = infer(words_to_template_mfcc_feats, input_fpath)
  print(f"Label: {label}")
  print(f"Predicted word: {predicted_word}")
  print(f"\tCost: {cost}")
  labels.append(label)
  predictions.append(predicted_word)

print(f"\nLabels: {labels}")
print(f"Predictions: {predictions}")
print(f"Accuracy: {accuracy_score(labels, predictions)}")