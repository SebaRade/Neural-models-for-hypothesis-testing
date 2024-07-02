import keras
from keras.utils import to_categorical
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import keras.backend as kb
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import pandas as pd

# Subsampling/shuffling for training and testing
np.random.seed(123)
total_samples = 280000
sub_samples = 70000

subset_idx = np.random.choice(total_samples, sub_samples)
X_subset, y_nt_subset, y_asd_subset = X_mixed[subset_idx], y_mixed[subset_idx], y_asd[subset_idx]

y_nt_subset = to_categorical(y_nt_subset)
y_asd_subset = to_categorical(y_asd_subset)
print(y_nt_subset.shape)
print(y_asd_subset.shape)

# Neurotypical CNN model
model_nt = Sequential([
  Conv2D(1, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(2, kernel_size=(3, 3), padding="same", activation="relu"),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(96, activation="relu"),
  Dense(64, activation="sigmoid", name="CA3"),
  Dense(96, activation="relu"),
  Dense(10, activation="softmax")])
optimizer_nt = keras.optimizers.RMSprop(learning_rate=0.0001)
model_nt.compile(optimizer=optimizer_nt, loss="categorical_crossentropy")

# ASD CNN model
model_asd = Sequential([
  Conv2D(1, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(2, kernel_size=(3, 3), padding="same", activation="relu"),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(96, activation="relu"),
  Dense(64, activation="sigmoid", name="CA3"),
  Dense(96, activation="relu"),
  Dense(10, activation="softmax")])
optimizer_asd = keras.optimizers.RMSprop(learning_rate=0.0001)
model_asd.compile(optimizer=optimizer_asd, loss="categorical_crossentropy")

# Synaptic plasticity CNN model
model_syn = Sequential([
  Conv2D(1, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(2, kernel_size=(3, 3), padding="same", activation="relu"),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(96, activation="relu"),
  Dense(64, activation="sigmoid", name="CA3"),
  Dense(96, activation="relu"),
  Dense(10, activation="softmax")])
optimizer_syn = keras.optimizers.RMSprop(learning_rate=0.00001)
model_syn.compile(optimizer=optimizer_syn, loss="categorical_crossentropy")

# Synaptic homeostasis CNN model
model_hom = Sequential([
  Conv2D(1, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(2, kernel_size=(3, 3), padding="same", activation="relu"),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(96, activation="relu"),
  Dense(64, activation="sigmoid", name="CA3"),
  Dense(96, activation="relu"),
  Dense(10, activation="softmax")])
optimizer_hom = keras.optimizers.RMSprop(learning_rate=0.0001, weight_decay=-.5)
model_hom.compile(optimizer=optimizer_hom, loss="categorical_crossentropy")

# Train models
hist_nt = model_nt.fit(X_subset.reshape(-1, 28, 28, 1), y_nt_subset, validation_split=0.33, epochs=10, batch_size=32, verbose=0)
hist_asd = model_asd.fit(X_subset.reshape(-1, 28, 28, 1), y_asd_subset, validation_split=0.33, epochs=10, batch_size=32, verbose=0)
hist_syn = model_syn.fit(X_subset.reshape(-1, 28, 28, 1), y_nt_subset, validation_split=0.33, epochs=10, batch_size=32, verbose=0)
hist_hom = model_hom.fit(X_subset.reshape(-1, 28, 28, 1), y_nt_subset, validation_split=0.33, epochs=10, batch_size=32, verbose=0)

# Plot loss vs. epochs
def losses(model, name):
  plt.figure(figsize=(5,5))
  plt.plot(model.history['loss'], color='#37abc8')
  plt.plot(model.history['val_loss'], color='#fba238')
  plt.title(name)
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper right')
  return plt

losses(hist_nt, "NT")
losses(hist_asd, "ASD")
losses(hist_syn, "SYN")
losses(hist_hom, "HOM")

# Run predictions and calculate accuracies
def pred(model, dataset, labels):
    pred = model.predict(dataset.reshape(-1, 28, 28, 1))
    pred = np.argmax(pred, axis=1)
    acc = accuracy_score(pred, labels.astype(int))
    return acc

datasets = [X, X_10, X_20, X_30, X_40, X_50, X_60, X_70, X_80, X_90, X_100]
labels = [y, y_10, y_20, y_30, y_40, y_50, y_60, y_70, y_80, y_90, y_100]

res_nt = [pred(model_base, i, j) for i, j in zip(datasets, labels)]
res_asd = [pred(model_base, i, j) for i, j in zip(datasets, labels)]
res_syn = [pred(model_syn, i, j) for i, j in zip(datasets, labels)]
res_hom = [pred(model_hom, i, j) for i, j in zip(datasets, labels)]

# Plot accuracies
noise = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
accuracies = pd.DataFrame(np.column_stack([noise, res_nt, res_asd, res_syn, res_hom]), columns=['noise', 'NT', 'ASD', 'SYN', 'HOM']).set_index('noise')
accuracies = accuracies.melt(ignore_index=False).reset_index()
accuracies.rename(columns={'variable': 'model', 'value': 'accuracy'}, inplace=True)

sns.pointplot(data=accuracies, x='noise', y='accuracy', hue='model', palette=['#394754', '#006680', '#37abc8', '#fba238'])
plt.ylabel('Accuracy')
plt.xlabel('Noise')
plt.ylim(0,1)

# Get weights of "CA3" layer
w_nt = np.mean(model_nt.get_layer('CA3').get_weights()[0], axis=1)
w_asd = np.mean(model_asd.get_layer('CA3').get_weights()[0], axis=1)
w_syn = np.mean(model_syn.get_layer('CA3').get_weights()[0], axis=1)
w_hom = np.mean(model_hom.get_layer('CA3').get_weights()[0], axis=1)
weights = [w_nt, w_asd, w_syn, w_hom]

# Plot weights in a violin plot (according to https://medium.com/@alexbelengeanu/getting-started-with-raincloud-plots-in-python-2ea5c2d01c11)
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#fba238', '#37abc8', '#006680', '#394754']
bp = ax.boxplot(weights, patch_artist = True, vert = False)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(.2)
vp = ax.violinplot(weights, points=500, showmeans=False, showextrema=False, showmedians=False, vert=False)
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    b.set_color(colors[idx])
for idx, features in enumerate(weights):
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(features, y, s=.3, c=colors[idx])
plt.yticks(np.arange(1,5,1), ['HOM', 'SYN', 'ASD', 'NT'])
