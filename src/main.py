import glob
import os
from sklearn.metrics import accuracy_score

from feature_extractor import extract_mfcc_feats
from inference import infer

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
TEMPLATE_NAME = "sp-1"

fpaths = [os.path.abspath(fpath) for fpath in glob.glob(os.path.join(DATA_DIR, TEMPLATE_NAME, "*.wav"))]
words_to_template_mfcc_feats = {
  fpath.split('/')[-1].split('.')[0].lower(): extract_mfcc_feats(fpath) for fpath in fpaths
}

input_fpaths = [os.path.abspath(fpath) for fpath in glob.glob(os.path.join(DATA_DIR, "**", "*.wav")) if not fpath.startswith(os.path.join(DATA_DIR, TEMPLATE_NAME))]
input_fpaths_to_label_and_mfcc_feats = {
  input_fpath: (input_fpath.split('/')[-1].split('.')[0].lower(), extract_mfcc_feats(input_fpath)) for input_fpath in input_fpaths
}

labels, predictions = [], []
for i, input_fpath in enumerate(input_fpaths_to_label_and_mfcc_feats) :
  print(i+1, input_fpath)
  label = input_fpaths_to_label_and_mfcc_feats[input_fpath][0]
  predicted_word, cost, path = infer(
    words_to_template_mfcc_feats,
    input_fpath,
    mode="euclidean"
  )
  print(f"\tLabel: {label}")
  print(f"\tPredicted word: {predicted_word}")
  print(f"\tCost: {cost}")
  labels.append(label)
  predictions.append(predicted_word)

print(f"\nLabels: {labels}")
print(f"Predictions: {predictions}")
print(f"Accuracy: {accuracy_score(labels, predictions)}")