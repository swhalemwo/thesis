from clickhouse_driver import Client
import numpy as np
import pandas as pd
import csv
from graph_tool.all import *
from graph_tool import *
import matplotlib.pyplot as plt
import random
import time
import itertools
import copy
from collections import Counter
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

mat_song_smpl = load_npz(diag_dir +'mats/mat_song_smpl.npz')
mat_edge_smpl = load_npz(diag_dir + 'mats/mat_edge_smpl.npz')
mat_cutofs = load_npz(diag_dir + 'mats/mat_cutofs.npz')

mat_dict = {'mat_song_smpl':mat_song_smpl, 'mat_edge_smpl':mat_edge_smpl, 'mat_cutofs':mat_cutofs}

# mats = ['mat_song_smpl','mat_edge_smpl','mat_cutofs']
mats = ['mat_song_smpl']
comps = [3,4,5]
max_iters = [30, 37, 45]
doc_topic_priors = [0.1, 0.25, 0.4]
topic_word_priors = [0.25, 0.4, 0.55, 0.7]

configs = []
for mat in mats:
    for comp in comps:
        for max_iter in max_iters:
            for alpha in doc_topic_priors:
                for beta in topic_word_priors:
                    config = {'mat' :mat, 'max_iter':max_iter, 'doc_topic_prior':alpha, 'topic_word_prior':beta,
                              'n_components':comp}
                    configs.append(config)


# ** running

for cfg in configs:
    for itr in range(3,5):

        t1 = time.time()
        ldax = LatentDirichletAllocation(n_jobs = 3,
                                         n_components=cfg['n_components'], 
                                         max_iter = cfg['max_iter'],
                                         doc_topic_prior = cfg['doc_topic_prior'],
                                         topic_word_prior = cfg['topic_word_prior']
        )
        ldax.fit(mat_dict[cfg['mat']])
        t2 = time.time()
        print(cfg, t2-t1)
    
        flnm = cfg['mat'] + "_iter_" + str(cfg['max_iter']) + "_alpha_" + str(cfg['doc_topic_prior']) + "_beta_" + str(cfg['topic_word_prior']) + '_ncomp' + str(cfg['n_components']) + '_it_' + str(itr)

        dump(ldax, diag_dir + flnm)

# mat_edge_smpl: one iterations takes 1.22 secs
# mat_song_smpl: one iteration takes 3.6 secs
# mat_cutofs: 3.9 secs
# on average 2.9 secs
# on average 35 iterations
# total of 135*35 = 4725 iterations
# 4725*2.9 = 13702.5 secs = 3.8 hours


# xyz = load(diag_dir + flnm)

# * eval
# if cfg['mat'] == 'mat_song_smpl' and cfg['max_iter'] > 20:

res_dicts = []
mrbshp_dict = {}

for cfg in configs:

    for itr in range(1,5):
        flnm = cfg['mat'] + "_iter_" + str(cfg['max_iter']) + "_alpha_" + str(cfg['doc_topic_prior']) + "_beta_" + str(cfg['topic_word_prior']) + '_ncomp' + str(cfg['n_components']) + '_it_' + str(itr)
        print(flnm)
        
        resx = load(diag_dir + flnm)
        scr = resx.score(mat_dict[cfg['mat']])

        mbrshp = resx.transform(mat_dict[cfg['mat']])
        mrbshp_dict[flnm] = mbrshp
        cls_nice = len(np.where(mbrshp > 0.70)[0])/(mbrshp.shape[0])

        res_dict = copy.deepcopy(cfg)
        res_dict['itr'] = itr
        res_dict['scr'] = scr
        res_dict['cls_nice'] = cls_nice

        res_dicts.append(res_dict)
        print(res_dict)


# res_df_cutofs = res_df
# res_df_cutofs.to_csv('res_df_cutofs.csv')



res_df = pd.DataFrame(res_dicts)

res_df = res_df[res_df.n_components == 5]
# res_df = res_df[(res_df.mat == 'mat_cutofs') & (res_df.max_iter > 20)]

res_df = df_rel

res_df = res_df[res_df.ptn == 5]

grp_vrbl = 'topic_word_prior' 
x_vrbl = 'max_iter'
# y_vrbl = 'cls_nice'
y_vrbl = 'ptn_sz_sd'

res_sum = res_df[[grp_vrbl, x_vrbl, y_vrbl]].groupby([grp_vrbl, x_vrbl]).mean()

res_df[[grp_vrbl, x_vrbl, y_vrbl]].groupby([grp_vrbl]).mean()
res_df[[grp_vrbl, x_vrbl, y_vrbl]].groupby([grp_vrbl]).std()

res_plt = {}
for i in np.unique(res_df[grp_vrbl]):
    res_plt[i] = []

for r in res_sum.itertuples():
    res_plt[r.Index[0]].append(r[1])

plt.figure(figsize=(12, 8))
for i in res_plt.keys():
    plt.plot([str(i) for i in np.unique(res_df[x_vrbl])], res_plt[i], label = i)

plt.title("Choosing Optimal LDA Model")
plt.xlabel(x_vrbl)
plt.ylabel(y_vrbl)
plt.legend(title=grp_vrbl, loc='best')
plt.show()

# *** reliability results
# more iterations seem to increase  reliability
# not in really wrt to ptn_sz_sd tho, at least not across the board
# for ovrl_sim somewhat, for ptn_sz_sd2 quite broadly

# ptn_sz_sd: doc_topic_prior 0.1, beta 0.7 best
# but fit for that is pretty bad
# otherwise good one (alpha 0.1, beta 0.4) is medium in ptn_sz_sd (also medium for 3 and 4 clusters)
# ovrl sim: also alpha 0.1, beta 0.7 best

# 4 components: alpha 0.25, beta  0.7 best
# 3 components: alpha doesn't matter so much, beta 0.25 and 0.55 best

# beta 0.4 is worst for 5 ptns 

# runs with more iterations needed


# *** results wrt run 4
# 3 topics: alpha 0.1, beta 0.4
# 4 topics: alpha 0.25, beta 0.4 HUH alpha 0.1 super bad 
# 5 topics: alpha 0.1, beta 0.4-0.55
# i think i don't like 4 because of the symmetry
# i think more also fit better to method, and 5 is better compromise: still few but not too few
# results for 4 are different: alphas can be quite high
# 5 also allows neat numbers: default alpha and beta are 0.2, get adjusted to 0.1/0.4


# mats: edge sampling fits best, mat_cutofs worst, mat_song_smpl in between
# -> mat fit is proportional to the amount of information
# tbh i think with 10k songs there is still sufficient variation in popularity to use hard cutoffs
# -> use mat_cutofs

# with current mat, 50 does not add an improvement over 35 (70), 35 does over 20 tho (20k)


# topic_word_priors over 1 seem much worse
# inverse relationship?
# beta 0.4 is good at 0.15 alpha, but also at 0.5 alpha

# doc_topic prior: small ones clearly preferable: 0.15 best
# should compare sharpness of user clustering

# does beta even matter so much? just want the clustering
# pretty sure it does matter tho: has to be clear what topics people belong to

# doc_topic_prior 0.1, topic_word_prior 0.4 doesn't sound that bad

# * look at sharpness of user distribution
mat = 'mat_cutofs'
max_iter = 50
doc_topic_prior = 0.15
topic_word_prior= 1.2

matx = mat_dict[mat]

flnm = mat + "_iter_" + str(max_iter) + "_alpha_" + str(doc_topic_prior) + "_beta_" + str(topic_word_prior) + '_it_' + str(itr)
print(flnm)

resx = load(diag_dir + flnm)
scr = resx.score(mat_dict[cfg['mat']])

mbrshp = resx.transform(mat_dict[mat])
nph(mbrshp)

# hmm
# alpha 0.15, beta 0.4 not that amazing: only handful people clearly belonging to one style
# even worse tho for alpha 0.5, beta 0.4: only few hundreds have clear positive membership

# cls_nice: increases with higher beta, decreases with higher alpha
# but high nice_cls at betas > 1, where score decreases
# but still it's a difference of like 5 pct, only barely visible

after lunch: rework general function, run more simulations tonight

# * write back to CH

# test playcount

usr_plcnts_org = []
usr_plcnts_cutofs = []
usr_plcnts_e_smpl = []
usr_plcnts_s_smpl = []

c = 0
for u in unq_usrs:
    usr_plcnts_org.append(g_usrs.vertex(g_usrs_vd[u]).out_degree())
    
    usr_plcnts_cutofs.append(np.sum(mat_cutofs[c]))
    usr_plcnts_e_smpl.append(np.sum(mat_edge_smpl[c]))
    usr_plcnts_s_smpl.append(np.sum(mat_song_smpl[c]))
    c+=1
    if c % 25 == 0: print(c)
    
    
cnt_ar = np.array([usr_plcnts_org, usr_plcnts_cutofs, usr_plcnts_e_smpl, usr_plcnts_s_smpl])
x = np.corrcoef(cnt_ar)
plt.matshow(x)
plt.show()

# edge sampling really shit at reproducing original distribution, but the other two aren't great either with ~0.5
# but they themselves are very similar -> seem to do the same thing basically
# maybe use s_smpl? has slightly higher cutoff, and is more realistic; doesn't punish unpopular ones so bad
# also has better fit hmmm

# results are same tho overall: alpha 0.15, beta 0.4

res_df2 = res_df.groupby(['max_iter', 'doc_topic_prior', 'topic_word_prior']).mean()


cfg_use = res_df2[res_df2.scr == max(res_df2.scr)]


cfg_use['mat']
flnm = 'mat_song_smpl_iter_35_alpha_0.15_beta_0.4_it_2'
resx = load(diag_dir+ flnm)

mrbshp = resx.transform(mat_song_smpl)

client = Client(host='localhost', password='anudora', database='frrl')

# is easier with separate table

ptn_str = """CREATE table ptn (usr String, 
""" + ', '.join(['ptn' + str(i) + ' Float32' for i in range(mrbsh.shape[1])]) + """,
 rndm Int8) ENGINE = MergeTree()
PARTITION BY rndm
ORDER BY tuple()"""

client.execute(ptn_str)


ptn_rows = []
for u in zip(unq_usrs, mbrshp):
    ptn_row = [u[0]] + list(u[1]) + sample(range(10), 1)
    ptn_rows.append(ptn_row)

client.execute('INSERT INTO ptn values', ptn_rows)

# * explore using song coefs
song_coefs = resx.components_

song_plcnt = np.sum(song_coefs, axis = 0)
song_plcnt_ar = np.array([song_plcnt]*5)

song_rel_coef = song_coefs/song_plcnt_ar

high_coefs = np.where(song_rel_coef > 0.7)
# 60-75% of songs are nicely classified
nph(song_rel_coef)

# FUUU
# such convenient partitioning of the songs, would be pretty easy to add the other information
# also nice with absolute weights


# but might not be so bad if all the information is also in the user scores
# is it possible to get the components from the user scores?
mrbshp: 3212, 5
mat_song_smpl: 3121, 10k

components: 10k, 5

# pretty sure any two can be transformed into the other with matrix algebra
# need to check tho

# if so, then it's preferable to use the user scores because it allows to include more songs
# assumes that the used songs suffice to cluster
# it's a reach to some extent tbh: if stuff gets added that is not in LDA, how can you assume it is differentiated by it
# i.e. if there are differences between the groups in genres that are based on songs not included in the LDA, the partitioning was not based on those genres, so can i really say the groups differ wrt ot them? 

# * reliability

from sklearn.cluster.bicluster import SpectralBiclustering

num_ress = []
for cfg in configs:

    mbrshp_ar = np.zeros((3212,0))

    ptn_cnt = cfg['n_components']
    unq_blks = list(range(ptn_cnt))


    for itr in range(1,5):
        flnm = cfg['mat'] + "_iter_" + str(cfg['max_iter']) + "_alpha_" + str(cfg['doc_topic_prior']) + "_beta_" + str(cfg['topic_word_prior']) + '_ncomp' + str(cfg['n_components']) + '_it_' + str(itr)

        m_ar1 = mrbshp_dict[flnm]
        mbrshp_ar = np.concatenate((mbrshp_ar, m_ar1), axis=1)

    res_ar = mbrshp_ar.T
    res_mat = np.corrcoef(res_ar)
    
    clust_mdl = SpectralBiclustering(n_clusters = ptn_cnt)
    clust1 = clust_mdl.fit(res_mat)

    col_lbls = clust1.column_labels_
    col_ord = [list(np.where(col_lbls ==i)[0]) for i in unq_blks]
    col_ord2 = list(itertools.chain.from_iterable(col_ord))

    res_mat2 = res_mat[col_ord2,:][:,col_ord2]
    
    # plt.matshow(res_mat2)
    # plt.show()
    print('res extraction')

    ptn_means = [] # similarity scores of cells in partition, the higher the better
    ptn_cell_cnts = [] # number of cells per partition, should be equal (sd =0)
    ptn_szs = []
    ptn_szs_sds = []

    for p in range(ptn_cnt):
        # print(p)

        c = 0
        ptn_cols = []
        for i in col_lbls:
            if i == p:
                ptn_cols.append(c)
            c +=1

        rel_cels = list(itertools.combinations(ptn_cols, 2))
        rel_vlus = [res_mat[i] for i in rel_cels]
        # print(np.mean(rel_vlus), len(rel_vlus))

        ptn_means.append(np.mean(rel_vlus))
        ptn_cell_cnts.append(len(rel_vlus))

        ptn_grps = np.where(col_lbls == p)[0]

        ptn_sums = np.sum(mbrshp_ar[:,ptn_grps], axis=0)
        
        ptn_sz = np.mean(ptn_sums)
        ptn_szs.append(ptn_sz)
        ptn_szs_sds.append(np.std(ptn_sums))


    ovrl_sim = np.average(ptn_means, weights = ptn_cell_cnts)
    ptn_sz_sd = np.std(list(Counter(col_lbls).values()))
    ptn_sz_sd2 = np.mean(ptn_szs_sds)

    num_res = [ptn_cnt, cfg['max_iter'], cfg['doc_topic_prior'], cfg['topic_word_prior']] + [ovrl_sim, ptn_sz_sd, ptn_sz_sd2]
    print(num_res)
    num_ress.append(num_res)
    # + ptn_szs


df_rel = pd.DataFrame(num_ress, columns = ['ptn', 'max_iter', 'doc_topic_prior', 'topic_word_prior', 'ovrl_sim', 'ptn_sz_sd', 'ptn_sz_sd2'])

df_rel.groupby('ptn').mean()

# * scrap

# perplexity correlates 0.9999 with log score -> only needs one
# plxt = resx.perplexity(mat_dict[cfg['mat']])
# res_dict['plxt'] = plxt
