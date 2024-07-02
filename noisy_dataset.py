from sklearn.datasets import fetch_openml
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import pandas as pd

# Get and normalise mnist_784 dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0
print(X.shape)
print(y.shape)

# Addition of Gaussian noise (0-100%) to images
def noise(factor):
    X_fac = [i - factor * np.random.normal(loc=0.0, scale=1.0, size=(784,)) for i in X]
    X_fac = np.clip(X_fac, 0.0, 1.0)
    y_fac = y
    return X_fac, y_fac

X_10, y_10 = noise(.1)
X_20, y_20 = noise(.2)
X_30, y_30 = noise(.3)
X_40, y_40 = noise(.4)
X_50, y_50 = noise(.5)
X_60, y_60 = noise(.6)
X_70, y_70 = noise(.7)
X_80, y_80 = noise(.8)
X_90, y_90 = noise(.9)
X_100, y_100 = noise(1)

# Visualisation of noisy images
images = [X[1], X_10[1], X_20[1], X_30[1], X_40[1], X_50[1], X_60[1], X_70[1], X_80[1], X_90[1], X_100[1]]
labels = [i/100 for i in range(0,110,10)]

num_row = 1
num_col = 11

fig, axes = plt.subplots(num_row, num_col)
sns.set(rc={'axes.facecolor':'#F2F2F2', 'figure.facecolor':'#F2F2F2'})
for i in range(num_col):
    ax = axes[i]
    ax.axis('off')
    ax.imshow(images[i].reshape(28,28), cmap='gray_r')
    ax.set_title('{}'.format(labels[i]))
plt.tight_layout()

# Combination of noiseless and and noisy images
X_mixed = np.concatenate((X, X_10, X_20, X_30), axis=0)
y_mixed = np.concatenate((y, y_10, y_20, y_30), axis=0)
print(X_mixed.shape)
print(y_mixed.shape)
