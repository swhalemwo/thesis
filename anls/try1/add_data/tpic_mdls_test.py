import numpy as np
import pandas as pd
import csv
from graph_tool.all import *
from graph_tool import *
import matplotlib.pyplot as plt
import random
import time
import itertools

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation

from scipy.sparse import save_npz, load_npz



from joblib import dump, load
# pickle results
# actually use joblib, seems to work better

# 3 matrices: always 3.2k users, vary number of songs
# actually vary sampling method with keeping sampling methods constant at 8k
# - edge sampling: 0.3, min 12 usrs, min plcnt 18 (gets it nicely to 10k songs)
# - cutoffs: at least 20 listeners, at least 93 playcount
# column (song) sampling

# reliability: each 5 times
diag_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/lda_diag/'

mat_song_smpl = load_npz(diag_dir +'mat_song_smpl.npz')
mat_edge_smpl = load_npz(diag_dir + 'mat_edge_smpl.npz')
mat_cutofs = load_npz(diag_dir + 'mat_cutofs.npz')

mat_dict = {'mat_song_smpl':mat_song_smpl, 'mat_edge_smpl':mat_edge_smpl, 'mat_cutofs':mat_cutofs}

mats = ['mat_song_smpl','mat_edge_smpl','mat_cutofs']
max_iters = [20, 35, 50]
doc_topic_priors = [0.15, 0.3, 0.5]
topic_word_priors = [0.1, 0.2, 0.4, 0.75, 1.2]

configs = []
for mat in mats:
    for max_iter in max_iters:
        for alpha in doc_topic_priors:
            for beta in topic_word_priors:
                config = {'mat' :mat, 'max_iter':max_iter, 'doc_topic_prior':alpha, 'topic_word_prior':beta}
                configs.append(config)


for cfg in configs:
    for itr in range(1,3):

        t1 = time.time()
        ldax = LatentDirichletAllocation(n_components=5, n_jobs = 3,
                                         max_iter = cfg['max_iter'],
                                         doc_topic_prior = cfg['doc_topic_prior'],
                                         topic_word_prior = cfg['topic_word_prior']
        )
        ldax.fit(mat_dict[cfg['mat']])
        t2 = time.time()
        print(cfg, t2-t1)
    
        flnm = cfg['mat'] + "_iter_" + str(cfg['max_iter']) + "_alpha_" + str(cfg['doc_topic_prior']) + "_beta_" + str(cfg['topic_word_prior']) + '_it_' + str(itr)

        dump(ldax, diag_dir + flnm)

# mat_edge_smpl: one iterations takes 1.22 secs
# mat_song_smpl: one iteration takes 3.6 secs
# mat_cutofs: 3.9 secs
# on average 2.9 secs
# on average 35 iterations
# total of 135*35 = 4725 iterations
# 4725*2.9 = 13702.5 secs = 3.8 hours


# xyz = load(diag_dir + flnm)



