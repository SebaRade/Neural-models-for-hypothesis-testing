import numpy as np
from numpy import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# SVM without noise
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

clf = SVC()
clf.fit(X_train,y_train)

score_0 = clf.score(X_test, y_test)
score_10 = clf.score(X_10, y_10)
score_20 = clf.score(X_20, y_20)
score_30 = clf.score(X_30, y_30)
score_40 = clf.score(X_40, y_40)
score_50 = clf.score(X_50, y_50)
score_60 = clf.score(X_60, y_60)
score_70 = clf.score(X_70, y_70)
score_80 = clf.score(X_80, y_80)
score_90 = clf.score(X_90, y_90)
score_100 = clf.score(X_100, y_100)

# SVM with noise
# Subsampling/shuffling images for training and testing
total_samples = 280000
sub_samples = 70000

subset_idx = np.random.choice(total_samples, sub_samples)
X_subset, y_subset = X_mixed[subset_idx], y_mixed[subset_idx]
print(X_subset.shape)
print(y_subset.shape)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_subset, y_subset, test_size=0.3, random_state=0, shuffle=True)

clf2 = SVC()
clf2.fit(X_train2,y_train2)

score2_0 = clf2.score(X_test, y_test)
score2_10 = clf2.score(X_10, y_10)
score2_20 = clf2.score(X_20, y_20)
score2_30 = clf2.score(X_30, y_30)
score2_40 = clf2.score(X_40, y_40)
score2_50 = clf2.score(X_50, y_50)
score2_60 = clf2.score(X_60, y_60)
score2_70 = clf2.score(X_70, y_70)
score2_80 = clf2.score(X_80, y_80)
score2_90 = clf2.score(X_90, y_90)
score2_100 = clf2.score(X_100, y_100)

# Plot accuracies
y_val = [score_0, score_10, score_20, score_30, score_40, score_50, score_60, score_70, score_80, score_90, score_100]
y_val2 = [score2_0, score2_10, score2_20, score2_30, score2_40, score2_50, score2_60, score2_70, score2_80, score2_90, score2_100]
x_val = [i/10 for i in range(0, 11)]

scores = pd.DataFrame({'Noise': x_val, 'w/_noise': y_val2, 'w/o_noise': y_val})

plt.figure(figsize=(5,5))
sns.set(rc={'axes.facecolor':'#F2F2F2', 'figure.facecolor':'#F2F2F2'})

g = sns.lineplot(data=scores, x='Noise', y='w/o_noise', marker='o', color='#37abc8')
sns.lineplot(data=scores, x='Noise', y='w/_noise', marker='o', color='#fba238')
g.set_yticks([0,.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
g.set_xticks([0,.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
plt.legend(['w/ noise', 'w/o noise'], loc='upper right')
plt.xlabel('Noise')
plt.ylabel('Accuracy')
