import matplotlib.pyplot as plt
from matplotlib import cm as cm
#import torch
import numpy as np
import sklearn.metrics.pairwise as met
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

exp_name = 'nonlocalnope'
#cmap = cm.get_cmap('Greens')
cmap = cm.get_cmap('YlGnBu_r')
#cmap = cm.get_cmap('Blues_r')
#cmap = cm.get_cmap('magma')
#cmap = cm.get_cmap('RdYlGn')
tgtglob  = np.loadtxt('figure_'+exp_name+'/src_embedding.txt')
tgtglob2 = np.loadtxt('figure_'+exp_name+'/tgt_embedding.txt')
#120-140
tgtglob_similarity = np.abs(met.cosine_similarity(tgtglob[:,:], tgtglob[:,:]))
#tgtglob_similarity = tgtglob_similarity.reshape(1024*1024)
#tgtglob_similarity2 =np.abs(met.cosine_similarity(tgtglob2, tgtglob2))
#tgtglob_similarity2 = tgtglob_similarity2.reshape(1024*1024)
fig, ax = plt.subplots(figsize=(4,4))
p_index = 352
p_range = 30
#cax = ax.matshow(tgtglob_similarity[p_index:p_index+1,p_index-15:p_index+15], interpolation='nearest', cmap = cmap)
cax = ax.matshow(tgtglob_similarity, interpolation='nearest', cmap = cmap)
#plt.hist(tgtglob_similarity, 10, facecolor='blue', alpha = 0.5, label=r"w\o $\mathcal{L}_{geo}$")
#plt.hist(tgtglob_similarity2,num_bins, facecolor='red', alpha = 0.5, label=r"w\ $\mathcal{L}_{geo}$")
ax.grid(False)
#plt.title('San Francisco Similarity matrix')
plt.xticks([], [], rotation=90)
plt.yticks([], [])
#fig.colorbar(cax, ticks=[0, 0.2, 0.4, 0.6, 0.8,1 ])
plt.show()
#print(np.sum(np.diag(tgtglob_similarity)))
