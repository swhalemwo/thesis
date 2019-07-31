from graph_tool.all import *
from graph_tool import *

from sklearn.cluster.bicluster import SpectralBiclustering
import itertools
import time
import csv
import random
import argparse
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


if __name__ == '__main__':

    print('setup')
    basedir = "/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/diag/"

    num_res_file = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/diag/diag_res.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('smpl_prop', help='proportion of songs to select of user')
    parser.add_argument('sim_cutof', help='how similar two users have to be to be connected')
    parser.add_argument('ptn_cnt', help = 'number of partitions')

    args = parser.parse_args()
    smpl_prop = float(args.smpl_prop)
    sim_cutof = float(args.sim_cutof)
    ptn_cnt = int(args.ptn_cnt)

    # smpl_prop = 0.15
    # sim_cutof = 0.02
    # ptn_cnt = 4


    print('start consistency iterations')

    csist_res = []
    ts = []
    v_cnts = []
    e_cnts = []

    for it in range(1,4):
        
        flnm = "diag_onmd_smpl_" + str(smpl_prop) + "_sim_"+   str(sim_cutof) + '_it_' + str(it) + '.gt'
        g_one_mode = Graph()
        g_one_mode.load(basedir+ 'graphs/' +flnm)

        v_cnts.append(len(set(g_one_mode.vertices())))
        e_cnts.append(len(set(g_one_mode.edges())))

        t1 = time.time()

        state = minimize_blockmodel_dl(g_one_mode,
                                       B_min = ptn_cnt, B_max = ptn_cnt,
                                       deg_corr = True)

        t2 = time.time()
        print(t2-t1, it)
        ts.append(t2-t1)

        blks = state.get_blocks()
        blks_vlus = [blks[i] for i in g_one_mode.vertices()]
        print(Counter(blks_vlus))

        unq_blks = np.unique(blks_vlus)
        run_res = []

        for k in unq_blks: run_res.append([])

        c = 0
        for x in blks_vlus:
            run_res[x].append(g_one_mode.vp.id[g_one_mode.vertex(c)])
            c+=1

        csist_res.append(run_res)

    all_grps_len = len(list(itertools.chain.from_iterable(csist_res)))
    all_grps = list(itertools.chain.from_iterable(csist_res))

    res_mat = np.zeros((all_grps_len,all_grps_len))

    c1 = 0
    for i1 in all_grps:
        c2 = 0
        for i2 in all_grps:
            ovlp = 2*len(set(i1) & set(i2))/(len(set(i1)) + len(set(i2)))
            res_mat[c1,c2] = ovlp
            c2+=1

        c1 +=1


    clust_mdl = SpectralBiclustering(n_clusters = ptn_cnt)
    clust1 = clust_mdl.fit(res_mat)

    col_lbls = clust1.column_labels_
    col_ord = [list(np.where(col_lbls ==i)[0]) for i in unq_blks]
    col_ord2 = list(itertools.chain.from_iterable(col_ord))

    # row_lbls = clust1.row_labels_
    # row_ord = [list(np.where(row_lbls ==i)[0]) for i in unq_blks]
    # row_ord2 = list(itertools.chain.from_iterable(row_ord))

    # res_mat2 = res_mat[row_ord2,:][:,col_ord2]

    res_mat2 = res_mat[col_ord2,:][:,col_ord2]

    # plt.matshow(res_mat)
    # plt.matshow(res_mat2)
    # plt.show()


    # numerical info: std
    print('res extraction')

    ptn_means = []
    ptn_cell_cnts = []
    ptn_szs = []

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
        ptn_sz = np.mean([len(all_grps[i]) for i in ptn_grps])
        ptn_szs.append(ptn_sz)


    ovrl_sim = np.average(ptn_means, weights = ptn_cell_cnts)
    ptn_sz_sd = np.std(list(Counter(col_lbls).values()))

    nbr_es = np.mean(v_cnts)
    nbr_vs = np.mean(e_cnts)
    avg_t = np.mean(ts)

    num_res = [ptn_cnt, smpl_prop, sim_cutof, ovrl_sim, ptn_sz_sd, nbr_vs, nbr_es, avg_t] + ptn_szs

    with open(num_res_file, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow(num_res)

    mat_flnm = "ptn_" + str(ptn_cnt) + "_prop_" + str(smpl_prop) + "_ctf_" + str(sim_cutof)

    np.savetxt(basedir + 'mats/' + mat_flnm + "_org.csv", res_mat)
    np.savetxt(basedir + 'mats/' + mat_flnm + "_clst.csv", res_mat2)
    
    # res_mat_rest = np.genfromtxt(basedir + mat_flnm + "_clst.csv")
    

# * scrap
#     c1 = 0
# for i in csist_res:
#     for i2 in csist_res[csist_res.index(i)]:
#         c2 = 0
#         for k in csist_res:
#             for k2 in csist_res[csist_res.index(k)]:
#                 ovlp = 2*len(set(i2) & set(k2))/(len(set(i2)) + len(set(k2)))
#                 res_mat[c1,c2] = ovlp
#                 c2+=1

#         c1 +=1
