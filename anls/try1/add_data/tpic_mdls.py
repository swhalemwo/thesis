import numpy as np
import pandas as pd
import csv
from graph_tool.all import *
from graph_tool import *
import matplotlib.pyplot as plt
from random import sample
import time
import itertools

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation


# from sklearn.datasets import make_multilabel_classification
# X, _ = make_multilabel_classification(random_state=0)


def vd_fer(g, idx):
    """creates vertex index dict from graph and id property map"""
    # vertex_dict
    vd = {}
    vdrv = {}
    for i in g.vertices():
        vd[idx[i]] = int(i)
        vdrv[int(i)] = idx[i]
        
    return(vd, vdrv)


lnk_file = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/diag/usr_trk_lnks.csv'

with open(lnk_file, 'r') as fi:
    rd = csv.reader(fi)
    lnks = [(i[0], i[1], int(i[2])) for i in rd]

    

unq_usrs = np.unique([i[0] for i in lnks])

g_usrs = Graph()
g_usrs.edge_properties['plcnt'] = g_usrs.new_edge_property('int')

g_usrs.vp['id'] = g_usrs.add_edge_list(lnks, hashed = True, string_vals = True, eprops = [g_usrs.ep.plcnt])
g_usrs_vd, g_usrs_vd_rv = vd_fer(g_usrs, g_usrs.vp.id)

usr_ids = [g_usrs_vd[i] for i in unq_usrs]

# ** sampling
smpl_prop = 0.3
smpl_ep = g_usrs.new_edge_property('bool')
smpl_vp = g_usrs.new_vertex_property('bool')

usr_smpl = random.sample(set(unq_usrs), 3212)

for u in usr_smpl:

    vu = g_usrs.vertex(g_usrs_vd[u])
    smpl_vp[vu] = True
    u_dg_org = vu.out_degree()

    alctd_dg = u_dg_org*smpl_prop
    new_deg = 0

    sngs_el = list(vu.out_edges())
    random.shuffle(sngs_el)

    for e in sngs_el:
        new_deg = new_deg + 1
        smpl_ep[e] = True
        smpl_vp[e.target()] = True

        if new_deg > alctd_dg:
            break

g_usrs2 = Graph(GraphView(g_usrs, efilt = smpl_ep, vfilt = smpl_vp), prune=True)
g_usr_smpl_vd, g_usr_smpl_vd_rv = vd_fer(g_usrs2, g_usrs2.vp.id)

smpl_ids = [g_usr_smpl_vd[i] for i in usr_smpl]

ad_mat = graph_tool.spectral.adjacency(g_usrs2, weight = g_usrs2.ep.plcnt)
ad_mat2 = ad_mat[:,smpl_ids].T

col_nonzs = np.count_nonzero(ad_mat2.toarray(), axis=0)
rel_cols = np.where(col_nonzs > 12)[0]

ad_mat3 = ad_mat2[:,rel_cols]

col_sums = np.sum(ad_mat3, axis = 0)
rel_cols2 = np.where(col_sums > 18)[1]

ad_mat4 = ad_mat2[:,rel_cols2]

# save_npz(diag_dir + 'mat_edge_smpl.csv', ad_mat4)


# ** no sampling

ad_mat = graph_tool.spectral.adjacency(g_usrs, weight = g_usrs.ep.plcnt)
ad_mat2 = ad_mat[:,usr_ids].T

col_nonzs = np.count_nonzero(ad_mat2.toarray(), axis=0)
rel_cols = np.where(col_nonzs > 15)[0]

ad_mat3 = ad_mat2[:,rel_cols]

col_sums = np.sum(ad_mat3, axis = 0)
rel_cols2 = np.where(col_sums > 50)[1]

ad_mat4 = ad_mat3[:,rel_cols2]

# save_npz(diag_dir + 'mat_cutofs', ad_mat4)

ad_mat5 = ad_mat4[:,sample(range(ad_mat4.shape[1]), 10100)]
save_npz(diag_dir + 'mat_song_smpl', ad_mat5)


# * lda

# takes around 1min for 3.2k usrs adn 100k songs, not bad at all

lda = LatentDirichletAllocation(n_components=4,random_state=0, n_jobs = 3, max_iter = 100)

t1 = time.time()
lda_res = lda.fit(ad_mat4)
t2 = time.time()

# 1k users, 9.5k songs: 8.2 secs
# 1k usrs, 18k songs: 8.34 secs
# 2k usrs, 22k songs: 15 secs
# 2k usrs, 5.4k songs: 11 secs
# 2k usrs, 41k songs: 17 secs
# probably matters how important songs are
# 2k usrs, 93k songs, smple_prop 1, min sum 5: 54 secs
# 2k usrs: 13k songs, smpl_prop 1, min_sum 50: 31 sec
# 2k usrs: 26k songs, smpl_propp 1, min_sum 30: 39 secs
# 2k usrs: 5k songs, smpl_propp 1, min_sum 100: 16 secs

# add iter changes:
# https://stackoverflow.com/questions/15067734/lda-model-generates-different-topics-everytime-i-train-on-the-same-corpus/15069580#15069580 says it's increases consistency
# 2k usrs: 5k songs, smpl_propp 1, min_sum 100, max_iter = 20: 24 secs
# 2k usrs: 5k songs, smpl_propp 1, min_sum 100, max_iter : 30: 35 secs
# 2k usrs: 5k songs, smpl_propp 1, min_sum 100, max_iter : 50: 47 secs, -7342576.182129103
# 2k usrs: 5k songs, smpl_propp 1, min_sum 100, max_iter: 10: 47 secs, 16 sec, -7372235.348836743

lda_res.components_[0:5,0:5]

lda_scrs = lda.score(ad_mat4)

usr_grp = lda.transform(ad_mat4)


# search_params = {'n_components': [4, 6, 8], 'doc_topic_prior':[0.1, 0.2, 0.4, 0.8, 2, 4], 'topic_word_prior':[0.1, 0.2, 0.4, 0.8, 2, 4]}

# search_params = {'n_components': [2,3,4], 'topic_word_prior':[0.1, 0.8, 2, 4, 8]}

# search_params = {'n_components': [2,3], 'topic_word_prior':[0.1, 2, 8]}

# search_params = {'max_iter': [10, 20, 30, 40, 50, 60]}

#  test different impact of alpha and beta conditional on number of topics
# search_params = {'doc_topic_prior': [0.033, 0.1, 0.4, 0.8, 1.2, 2], 'topic_word_prior': [0.033, 0.1, 0.4, 0.8, 1.2, 2]}


# search_params = {'topic_word_prior': [0.033, 0.1, 0.4, 0.8, 1.2, 2]}
# search_params = {'topic_word_prior': [0.1, 0.25, 0.4, 0.55, 0.7]}

search_params = {'topic_word_prior': [0.1, 0.175, 0.25, 0.325, 0.4, 0.475, 0.55]}
search_params = {'doc_topic_prior': [0.1, 0.175, 0.25, 0.325, 0.4, 0.475, 0.55]}

search_params = {'max_iter':[10, 30, 40, 50, 75, 100]}

search_params = {'max_iter':[10, 15, 20, 25, 30, 10, 15, 20, 25, 30, 10, 15, 20, 25, 30]}

# topic_word_prior = 0.2
lda = LatentDirichletAllocation(learning_method='batch', n_jobs = 4, n_components = 5, max_iter = 30, topic_word_prior = 0.45)
model = GridSearchCV(lda, param_grid=search_params, verbose=2, cv =[(slice(None), slice(None))],
                     return_train_score = True, refit = False)
model.fit(ad_mat4)

npl(model.cv_results_['mean_test_score'])

# ordr = [0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4,  9, 14]
# npl(model.cv_results_['mean_test_score'][ordr])


# 1k people, 1 sampling, min plcnt 100 -> 1.6k songs: looks like there is no systematic increase after 30 iterations? 
# test with more people
# 3.2k people, full sampling, 24k songs, min plcnt 50: 40 best, largest improvement by going from 10 to 20

best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(ad_mat4))

n_topics = [4,6,8]

res_df_prep = []

for i in model.cv_results_:
    print(i)

for i in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
    i[0]['score'] = i[1]
    res_df_prep.append(i[0])
    
res_df = pd.DataFrame(res_df_prep)


# grp_vrbl = 'n_components'
grp_vrbl = 'topic_word_prior'
# x_vrbl = 'doc_topic_prior'
x_vrbl = 'doc_topic_prior'

res_sum = res_df[[grp_vrbl, x_vrbl, 'score']].groupby([grp_vrbl, x_vrbl]).mean()

res_plt = {}
for i in np.unique(res_df[grp_vrbl]):
    res_plt[i] = []

for r in res_sum.itertuples():
    res_plt[r.Index[0]].append(r.score)

    
# Show graph
plt.figure(figsize=(12, 8))
for i in res_plt.keys():
    plt.plot([str(i) for i in np.unique(res_df[x_vrbl])], res_plt[i], label = i)

# plt.plot(n_topics, log_likelyhoods_5, label='0.5')
# plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.title("Choosing Optimal LDA Model")
plt.xlabel(x_vrbl)
plt.ylabel("Log Likelyhood Scores")
plt.legend(title=grp_vrbl, loc='best')
plt.show()

# doc_topic_prior has no much impact, can be left default i guess
# topic_word_prior has quite some impact, should go even higher -> 0.1, 0.8, 2, 4, 8
# less topics -> better: test difference between 2,3,4


# relevant parameters:
- number of topics
- doc_topic_prior: alpha: default 1/nbr_components: prior of document topic distribtuion
- topic_word_prior: beta/eta: default 1/nbr_components: prior of tpic word distribution
Close to 1 seems to represent lack of knowledge:
distribution in middle; otherwise in the corners -> vary between 0.1, 0.2, 0.4, 0.8, 2, 4

- learning method: online faster? 
  - learning decay: only relevant in online: use later

- number of iterations

# reduce time with less songs: focus on 10k most used ones
# sampling songs: have to make sure that's reflected in the adjacency matrix
# ok looks actually pretty decently fast with proper matrix



# * reliability
# get the score colums (transform): show show clusters

# also check what this k-fold validation exactly does
# also wrt scores

# is there a similar basic thing that does what i'm trying to do?
# precision, recall?

# gridsearchdv re-uses code and is therefore faster -> would be preferable to use

# impact of max_iter
# impact of usrs and songs: what is more expensive? 


lda2 = LatentDirichletAllocation(n_components=4, n_jobs = 4, topic_word_prior = 0.45, max_iter = 50)
# online seems slower? 109 vs 88

res_ar = np.empty((ad_mat4.shape[0],0))

for i in range(4):
    t1 = time.time()
    lda_res = lda2.fit(ad_mat4)
    t2 = time.time()
    print(t2-t1)

    usr_grp = lda2.transform(ad_mat4)

    res_ar = np.concatenate((res_ar, usr_grp), axis=1)


cor_mat = np.corrcoef(res_ar.T)
cor_mat = cosine_similarity(res_ar.T)
plt.matshow(cor_mat)
plt.show()

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.metrics.pairwise import cosine_similarity

clust_mdl = SpectralBiclustering(n_clusters = 4)
clust1 = clust_mdl.fit(cor_mat)

col_lbls = clust1.column_labels_
col_ord = [list(np.where(col_lbls ==i)[0]) for i in list(range(4))]
col_ord2 = list(itertools.chain.from_iterable(col_ord))

res_mat2 = cor_mat[col_ord2,:][:,col_ord2]
plt.matshow(res_mat2)
plt.show()

# res_mat_nol_iter = res_mat2

# oh well
# at least framework there
# 
# max_iter 50 not enough for reliability
# actually more iter worse? doesn't find topics
# wonder if there's a difference between correlation and cosine similarity? 

# * scrap
# works
# ad_mat3 =ad_mat2.toarray()
# u = unq_usrs[1312]
# sum(ad_mat3[1312])
# g_usrs.vertex(g_usrs_vd[u]).out_degree(g_usrs.ep.plcnt)


