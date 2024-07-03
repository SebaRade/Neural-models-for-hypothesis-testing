import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
import scikit_posthocs as sp

# Analyse accuracies from five replicates
a_cnn = pd.read_csv("accuracies_cnn.csv")

sns.pointplot(data=a_cnn, x='noise', y='acc', hue='model', palette=['#394754', '#006680', '#37abc8', '#fba238'])
plt.ylabel('Accuracy')
plt.xlabel('Noise')
plt.ylim(0,1)

# Compare distributions
def linmodel(model):
    """Calculates the logit per run and models the linear part using OLS

    Parameters
    ----------
    model : str
        The model of interest

    Returns
    -------
    DataFrame
        containing the slope (m) and intercept (b) from OLS model per run
    """

    runs = ['run1', 'run2', 'run3', 'run4', 'run5']
    res = []
    for run in runs:
        filt = a_cnn[(a_cnn['model'] == model) & (a_cnn['run'] == run)]
        Y = filt['acc'].apply(lambda x: math.log(x/(1-x)))
        X = filt['noise']
        X = sm.add_constant(X)
        ols = sm.OLS(Y,X)
        results = ols.fit()
        params = results.params
        res.append([params[1], params[0]])
    return pd.DataFrame(res, columns=['m', 'b'])

lin_nt = linmodel('model_nt')
lin_asd = linmodel('model_asd')
lin_syn = linmodel('model_syn')
lin_hom = linmodel('model_hom')

dunn_lin_slope = sp.posthoc_dunn([lin_nt['m'], lin_asd['m'], lin_syn['m'], lin_hom['m']], p_adjust='bonferroni')
dunn_lin_intercept = sp.posthoc_dunn([lin_nt['b'], lin_asd['b'], lin_syn['b'], lin_hom['b']], p_adjust='bonferroni')
print(dunn_lin_slope)
print(dunn_lin_intercept)

# Analyse weights from five replicates
w_cnn = pd.read_csv("weights_cnn.csv")

w_cnn_nt = np.array(w_cnn[w_cnn['model']=='nt']['weight'])
w_cnn_asd = np.array(w_cnn[w_cnn['model']=='model_asd']['weight'])
w_cnn_syn = np.array(w_cnn[w_cnn['model']=='model_syn']['weight'])
w_cnn_hom = np.array(w_cnn[w_cnn['model']=='model_hom']['weight'])
w_cnn2 = [w_cnn_hom, w_cnn_syn, w_cnn_asd, w_cnn_nt]

# Plot weights in a violin plot (according to https://medium.com/@alexbelengeanu/getting-started-with-raincloud-plots-in-python-2ea5c2d01c11)
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#fba238', '#37abc8', '#006680', '#394754']
bp = ax.boxplot(w_cnn2, patch_artist = True, vert = False)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(.2)
vp = ax.violinplot(w_cnn2, points=500, showmeans=False, showextrema=False, showmedians=False, vert=False)
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    b.set_color(colors[idx])
for idx, features in enumerate(w_cnn2):
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(features, y, s=.3, c=colors[idx])
plt.yticks(np.arange(1,5,1), ['HOM', 'SYN', 'ASD', 'NT'])

# Calculate weight means for comparison
w_cnn_nt2 = np.array(w_cnn[w_cnn['model']=='nt'].groupby(w_cnn['run'])['weight'].mean())
w_cnn_asd2 = np.array(w_cnn[w_cnn['model']=='model_asd'].groupby(w_cnn['run'])['weight'].mean())
w_cnn_syn2 = np.array(w_cnn[w_cnn['model']=='model_syn'].groupby(w_cnn['run'])['weight'].mean())
w_cnn_hom2 = np.array(w_cnn[w_cnn['model']=='model_hom'].groupby(w_cnn['run'])['weight'].mean())

dunn_w = sp.posthoc_dunn([w_cnn_nt2, w_cnn_asd2, w_cnn_syn2, w_cnn_hom2], p_adjust='bonferroni')
print(dunn_w)
