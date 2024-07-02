import numpy as np
from numpy import random

# Introduction of errors to mimic ASD conditions
def shuff():
  n = .5
  y_shuff = y.copy()
  ix = np.random.choice([True, False], size=y_shuff.size, replace=True, p=[n, 1-n])
  y_shuff_shuff = y_shuff[ix]
  np.random.shuffle(y_shuff_shuff)
  y_shuff[ix] = y_shuff_shuff
  return y_shuff

np.random.seed(123)

y_shuff_0 = shuff()
y_shuff_10 = shuff()
y_shuff_20 = shuff()
y_shuff_30 = shuff()

y_asd = np.concatenate((y_shuff_0, y_shuff_10, y_shuff_20, y_shuff_30), axis=0)
