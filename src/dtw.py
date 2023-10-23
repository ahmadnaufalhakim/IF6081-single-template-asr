from collections import defaultdict
import numbers
import numpy as np
from pprint import pprint
from scipy.spatial import distance
import types
import warnings

def minkowski(p:int) :
  return lambda v1, v2: distance.minkowski(v1, v2, p)

def validate_inputs(x:np.array, y:np.array, dist=None) :
  x = np.asanyarray(x, dtype="float")
  y = np.asanyarray(y, dtype="float")

  if x.shape[1] != y.shape[1] :
    raise ValueError("the number of MFCC features of x and y must be the same")
  if isinstance(dist, numbers.Number) :
    if dist <= 0 :
      raise ValueError("dist must be a positive integer")
    elif 0 < dist < 1 :
      warnings.warn("consider using a positive integer for distance (dist).\nNote: Some distance metrics may not satisfy the triangle inequality for dist < 1.")
  elif dist is not None and not isinstance(dist, types.FunctionType) :
    raise TypeError("dist must be of type function or integer")

  if dist is None :
    dist = minkowski(p=2)
  elif isinstance(dist, numbers.Number) :
    dist = minkowski(p=dist)
  return x, y, dist

def dtw(x:np.array, y:np.array, dist=None) :
  """returns the distance between 2 time series of MFCC features

  Parameters
  ----------
  x : array_like
    input MFCC array 1
  y : array_like
    template MFCC array 2
  dist : function or int
    The distance method used to calculate distance between x[i] and y[j].
    If dist is an int of value >0, then the Minkowski distance is used with p-value = dist
    If dist is a function, then distance is calculated with dist(x[i], y[j])
    If dist is None, then distance is calculated using Euclidean distance (Minkowski with p-value = 2)
  
  Returns
  -------
  distance : float
    the distance between 2 time series of MFCC features
  path : list
    list of indices for each input x and y
  """
  x, y, dist = validate_inputs(x, y, dist)
  # Get lengths of input and template sequences
  len_x, len_y = len(x), len(y)

  # Initialize a dictionary to store best path costs (P[i,j])
  # i-th input frame aligns with j-th template frame
  P = defaultdict(lambda: (float("inf"),))

  # Initialize the origin's path cost to 0
  P[0,0] = (0,0,0) # Note: P[i,j][0] is best cost, (P[i,j][1],P[i,j][2]) is the best path

  # Populate the P dictionary with all the best cost and paths for each (i,j) node in the
  # search space with size len_x*len_y while keeping track of the best last template index
  # for last input index
  best_last_j, best_cost = None, float("inf")
  for i in range(1, len_x+1) :
    for j in range(1, len_y+1) :
      cost = dist(x[i-1], y[j-1])
      # Determine which previous alignment node to choose
      if i == 1 :
        P[i,j] = min(
          (P[i-1,j-1][0] + cost, i-1, j-1),
          (P[i-1,j-2][0] + cost, i-1, j-2),
          (P[i-1,j-3][0] + cost, i-1, j-3),
          key= lambda tup: tup[0]
        )
      else :
        P[i,j] = min(
          (P[i-1,j][0] + cost, i-1, j),
          (P[i-1,j-1][0] + cost, i-1, j-1),
          (P[i-1,j-2][0] + cost, i-1, j-2),
          key= lambda tup: tup[0]
        )
        if i == len_x and P[i,j][0] < best_cost :
          best_last_j = j
          best_cost = P[i,j][0]

  # Reconstruct the best path
  best_path = []
  i, j = len_x, best_last_j
  while not (i == j == 0) :
    best_path.append((i-1, j-1))
    i, j = P[i,j][1], P[i,j][2]
  best_path.reverse()
  return best_cost, best_path

if __name__ == "__main__" :
  pass