import csv
import json
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
import operator
from datetime import datetime
from datetime import timedelta
from random import sample

from scipy.stats import entropy
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

from graph_tool.all import *
from graph_tool import *

from gnrl_funcs import get_dfs
from gnrl_funcs import dict_gnrgs
from gnrl_funcs import gini
from gnrl_funcs import weighted_avg_and_std

client = Client(host='localhost', password='anudora', database='frrl')

# * hierarchical relations based on split dimensions
# functions defined in item_tpclt

def vd_fer(g, idx):
    """creates vertex index dict from graph and id property map"""
    # vertex_dict
    vd = {}
    vdrv = {}
    for i in g.vertices():
        vd[idx[i]] = int(i)
        vdrv[int(i)] = idx[i]
        
    return(vd, vdrv)

def gnrt_acst_el(gnrs):
    """generates edge list for acoustic space network"""
    
    el_ttl = []
    
    for gnr in gnrs:
    #     print(gnr)
        
        # dfc = get_df_cbmd(gnr)
        # gnr_ind[gnr] = gnrz.index(gnr)
        dfcx = acst_gnr_dict[gnr]

        gnr_cnt = len(dfcx)
        sz_dict[gnr] = gnr_cnt
        
        el_gnr = []
        # print(len(el_gnr))

        wts = dfcx['rel_weight'] * dfcx['cnt']

        for vrbl in vrbls:

            bins = np.arange(0, 1.1, 0.1)
            a1, a0 = np.histogram(dfcx[vrbl], bins=10, weights=wts)
            # a_old, a0 = np.histogram(dfcx[vrbl], bins=10)

            nds1 = [vrbl + str(i) for i in range(1, len(bins))]
            nds2 = [gnr]*len(nds1)

            a_wtd = [i/sum(wts) for i in a1]
            # a_wtd_old = [i/gnr_cnt for i in a_old]
            a_wtd2 = [i/max(wts) for i in a1]

            elx = [i for i in zip(nds2, nds1, a1, a_wtd, a_wtd2)]

            el_gnr = el_gnr + elx

        el_ttl = el_ttl + el_gnr
        
    return(el_ttl)


def gnrt_acst_el_mp(gnrs):
    """parallelizes the acoustic edgelist generation process"""

    NO_CHUNKS = 3
    gnr_chnks = list(split(gnrs, NO_CHUNKS))

    t1 = time.time()
    p = Pool(processes=NO_CHUNKS)
    data = p.map(gnrt_acst_el, [i for i in gnr_chnks])
    t2=time.time()

    p.close()

    el_ttl = list(itertools.chain.from_iterable(data))
    return(el_ttl)



def gnrt_sup_dicts(acst_gnr_dict,gnrs):
    sz_dict = {}
    gnr_ind = {}

    for gnr in gnrs:
        sz_dict[gnr] = len(acst_gnr_dict[gnr])
        gnr_ind[gnr] = gnrs.index(gnr)
        
    return(sz_dict, gnr_ind)

# sz_dict, gnr_ind = gnrt_sup_dicts(acst_gnr_dict, gnrs)


def gac_crubgs(el_ttl):
    """constructs acoustic graph of genres and features"""
    
    gac = Graph()
    w = gac.new_edge_property('double')
    w_std = gac.new_edge_property('double')
    w_std2 = gac.new_edge_property('double')

    gac_id = gac.add_edge_list(el_ttl, hashed=True, string_vals=True,  eprops = [w, w_std, w_std2])

    vd,vdrv = vd_fer(gac, gac_id)
    return(gac, w, w_std, w_std2, gac_id, vd, vdrv)
    

# gt_sims = vertex_similarity(gac, 'dice', vertex_pairs = cmps, eweight=w_std2)


def sbst_eler(gt_sims, oprtr, thrshld):
    """assumes (asymmetric quadratic) similarity matrix as input"""

    # rel_cmps = np.where(gt_sims > thrshld)
    rel_cmps = np.where(oprtr(gt_sims, thrshld))

    rel_cmps2 = [i for i in zip(rel_cmps[0],rel_cmps[1])]

    el_acst = []

    for i in rel_cmps2:

        # direction is now from larger to more general 
        if i[0] == i[1]:
            continue
        
        else:
            vlu = gt_sims[i]
            cmp = [gnrs[i[1]], gnrs[i[0]], vlu]

        el_acst.append(cmp)
    return(el_acst)


# *** Graph asymmetric hierarchy acoustic
# assumes all relations beyond (above,under) thrshld are parent-subconcept relations

# el_acst = sbst_eler(asym_sim_ar, operator.gt, 0.99)

def kld_thrshld(el_acst):
    ghrac = Graph()
    sim_vlu = ghrac.new_edge_property('double')
    ghrac_id = ghrac.add_edge_list(el_acst, hashed=True, string_vals=True, eprops  = [sim_vlu])

    # vertex dict hierachical
    vd_hr, vd_hr_rv = vd_fer(ghrac, ghrac_id)

    # does not reliable capture birirectionality
    # graph_pltr(ghrac, ghrac_id, 'acst_spc3.pdf')

    # graph_draw(ghrac, output='ghrac.pdf')
    return(ghrac, sim_vlu, ghrac_id, vd_hr, vd_hr_rv)


def acst_arfy(el_ttl, vrbls, el_pos):

    acst_mat = np.empty([len(gnrs), len(vrbls)*10])

    c = ypos =xpos = 0

    for i in el_ttl:
        itempos = [ypos, xpos]
        # have to make sure to select right one here
        vlu = i[el_pos]
        # print(vlu)

        acst_mat[ypos, xpos] = vlu
        # print(itempos)

        xpos+=1

        if xpos == len(vrbls)*10:
            xpos = 0
            ypos+=1

        c+=1
    return(acst_mat)

# acst_mat = acst_arfy(el_ttl, vrbls,3)


# ** KLD
# needs functionizing and parallel processing 

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))



def kld_mp2(chnk):
    """ improved kdl with https://www.oipapio.com/question-4293090, no more inf fixing but shouldn't be needed it it's proper subsets, """
    # comparison with previous shows that places which are now also infinite are generally super high

    start_col = gnrs.index(chnk[0])
    end_col = gnrs.index(chnk[-1])+1

    kldx = entropy(acst_mat[start_col:end_col,:].T[:,:,None], acst_mat.T[:,None,:])
    return(kldx)
    


# t1 = time.time()
# for i in range(1000):
    # entropy(list(k_v)*10, list(i_v)*10)
    # entropy(list(k_v), list(i_v))
# t2 = time.time()
# increasing number of elements only slowly increases time (10*elements: twice as slow): logarithmic? 

def kld_mat_crubgs(gnrs):
    """multiproccesses acst_mat to create kld mat"""
    NO_CHUNKS = 3
    gnr_chnks = list(split(gnrs, NO_CHUNKS))

    p = Pool(processes=NO_CHUNKS)

    t1=time.time()
    data = p.map(kld_mp2, [i for i in gnr_chnks])
    t2=time.time()

    # t1=time.time()
    # data2 = p.map(kld_mp, [i for i in gnr_chnks])
    # t2=time.time()

    p.close()

    # ar_lst = [np.array(i) for i in data2]
    # ar_cb = np.concatenate(ar_lst, axis=0)

    ar_cb = np.concatenate(data, axis=0)

    return(ar_cb)

# need to deal with inf values
# where do they come from? 0s in A i think
# assess badness of fit: sum of cells of a that are 0 in b should not be above X (0.95)


# *** get N closest parents

def kld_n_prnts(ar_cb, npr):
    """generates edgelist by taking npr (number of parents) lowest values of row of asym kld mat"""

    kld2_el = []

    for i in gnrs:
        i_id = gnr_ind[i]
        sims = ar_cb[i_id]

        idx = np.argpartition(sims, npr)
        prnts = idx[0:4]
        vlus = sims[idx[0:4]]

        for k in zip(prnts, vlus):
            if k[0] == i_id:
                pass
            else:
                kld2_el.append((gnrs[k[0]], gnrs[i_id], k[1]))
                
    return(kld2_el)


def kld_proc(kld2_el):
    
    g_kld2 = Graph()
    kld_sim = g_kld2.new_edge_property('double')
    g_kld2_id = g_kld2.add_edge_list(kld2_el, hashed=True, string_vals=True, eprops = [kld_sim])

    vd_kld2,vd_kld2_rv = vd_fer(g_kld2, g_kld2_id)

    # get int e_vp for plotting
    # kld_sim_int = g_kld2.new_edge_property('int16_t')
    # for e in g_kld2.edges():
    #     kld_sim_int[e] = math.ceil(kld_sim[e]*100)

    # graph_pltr(g_kld2, g_kld2_id, 'acst_spc5.pdf', kld_sim_int)

    return(g_kld2, kld_sim, g_kld2_id, vd_kld2, vd_kld2_rv)


def all_cmps_crubgs(gnrs, vd, type):
    """constructs basic comparisons sets of nodes from different conditions"""
    # have to include combinations
    
    gnr_ids = [vd[i] for i in gnrs]
    lenx = len(gnrs)

    if type == "permutations":
        cmps = list(itertools.permutations(gnr_ids, 2))

    if type == "product":
        cmps = list(itertools.product(gnr_ids, repeat=2))
        
    return(cmps)




# * feature extraction


def ftr_extrct():
    """extracts features like a boss"""

    res_dict = {}
    for gnr in gnrs:
        res_dict[gnr] = {}

    cmps_rel, sim_v = gnr_span_prep(vrbls)

    for gnr in gnrs:
        # generate a whole bunch of measures

        gv = g_kld2.vertex(vd_kld2[gnr])

        # get sum of 3 distance to 3 parents
        prnt3_dvrg = gv.in_degree(kld_sim)

        clst_prnt = min([kld_sim[v] for v in gv.in_edges()])
        res_dict[gnr]['clst_prnt'] = clst_prnt

        res_dict[gnr]['prnt3_dvrg'] = prnt3_dvrg

        thd_res, thd_names = ar_cb_proc(gnr)
        for i in zip(thd_names, thd_res):
            # print(i)
            res_dict[gnr][i[0]] = i[1]

        # from original data, might be interesting to weigh/add sds
        res_dict[gnr]['sz_raw'] = sz_dict[gnr]
        res_dict[gnr]['avg_weight_rel'] = np.mean(acst_gnr_dict[gnr]['rel_weight'])

        cnt_x_rel_weight = sum(acst_gnr_dict[gnr]['rel_weight'] * acst_gnr_dict[gnr]['cnt'])
        res_dict[gnr]['cnt_x_rel_weight'] = cnt_x_rel_weight

        # get parents for all kinds of things
        prnts = list(g_kld2.vertex(vd_kld2[gnr]).in_neighbors())
        # outdegree of parents (weighted and unweighted)
        # may have to divide by 1 (or other thing to get distance), not quite clear now
        prnt_odg = np.mean([prnt.out_degree() for prnt in prnts])
        prnt_odg_wtd = np.mean([prnt.out_degree() * kld_sim[g_kld2.edge(prnt,gv)] for prnt in prnts])

        res_dict[gnr]['prnt_odg'] = prnt_odg
        res_dict[gnr]['prnt_odg_wtd'] = prnt_odg_wtd

        # cohorts
        cohrt_pct_inf, cohrt_mean_non_inf = chrt_proc(gnr)

        res_dict[gnr]['cohrt_pct_inf'] = cohrt_pct_inf
        res_dict[gnr]['cohrt_mean_non_inf'] = cohrt_mean_non_inf

        # spanningness

        spngns = gnr_mus_spc_spng(gnr, cmps_rel, sim_v)
        res_dict[gnr]['spngns'] = spngns

        dfcx_names, dfcx_vlus = dfcx_proc(gnr)
        for i in zip(dfcx_names, dfcx_vlus):
            res_dict[gnr][i[0]] = i[1]
        


    df_res = pd.DataFrame(res_dict).transpose()
    df_res['spngns_std'] = df_res['spngns']-min(df_res['spngns'])
    df_res['spngns_std'] = df_res['spngns_std']/max(df_res['spngns_std'])

    return(df_res)


## ** amount of musical space spanning
# use similar logic of omnivorousness

def gnr_span_prep(vrbls):
    """prepares feature similarity matrix, needed to see how well genres span"""
    # not sure if good:
    # weight
    
    vrbl_nd_strs_raw = [[vrbl + str(i) for i in range(1,11)] for vrbl in vrbls]
    vrbl_nd_strs = list(itertools.chain.from_iterable(vrbl_nd_strs_raw))

    vrbl_cmprs = all_cmps_crubgs(vrbl_nd_strs, vd, 'product')

    # vrbl_sims = vertex_similarity(GraphView(gac, reversed=True), 'dice', vertex_pairs = vrbl_cmprs, eweight = w_std)
    vrbl_sims = vertex_similarity(GraphView(gac, reversed=True), 'dice', vertex_pairs = vrbl_cmprs, eweight = w)
    vrbl_sim_rows = np.split(vrbl_sims, len(vrbl_nd_strs))
    vrbl_sim_ar = np.array(vrbl_sim_rows)

    # plt.imshow(1-vrbl_sim_ar, cmap='hot', interpolation='nearest')
    # plt.show()

    # sims or dsims? 

    vrbl_nds = [vd[i] for i in vrbl_nd_strs]

    # map to array row/col positions
    vrbl_mat_ids = {}
    for i in vrbl_nds:
        vrbl_mat_ids[i] = vrbl_nds.index(i) 

    cmps_rel = list(itertools.combinations(vrbl_nds, 2))

    sim_v = [1-vrbl_sim_ar[vrbl_mat_ids[i[0]],vrbl_mat_ids[i[1]]] for i in cmps_rel]
    
    return(cmps_rel, sim_v)

# cmps_rel, sim_v = gnr_span_prep(vrbls)


def gnr_mus_spc_spng(gnr, cmps_rel, sim_v):
    """calculates sum of dissimilarities for a gnr"""
    # relies on feature nodes always being returned in the same order so that sim_v applies across genres

    t1 = time.time()
    gv = gac.vertex(vd[gnr])
    g_es_raw = list(gv.out_edges())
    
    g_es = {}
    for i in g_es_raw:
        g_es[int(i.target())] = w_std[i]

    e1_v = []
    e2_v = []

    for i in cmps_rel:

        # e1 = w_std[gac.edge(gv, gac.vertex(i[0]))]
        # e2 = w_std[gac.edge(gv, gac.vertex(i[1]))]

        e1 = g_es[i[0]]
        e2 = g_es[i[1]]

        e1_v.append(e1)
        e2_v.append(e2)

    x = np.array(e1_v) * np.array(e2_v) * sim_v
    ttl_asim = sum(x)
    
    t2 = time.time()
    return(ttl_asim)


# average of similarities of indegrees
# superordinates

# spr_ord = list(g_kld2.vertex(vd_kld2[gnr]).in_neighbors())

# sim_vlu[ghrac.edge(gv, v)]
# sim_vlu[ghrac.edge(v, gv)]


# ** cohort processing
def chrt_proc(gnr):

    gv = g_kld2.vertex(vd_kld2[gnr])
    prnts = list(gv.in_neighbors())
    cohrts = [list(pr.out_neighbors()) for pr in prnts]

    cohrt_pcts_inf = []
    cohrt_means_non_inf = []

    for cht in cohrts:
        cht_dists = []
        for v_cp in cht:
            if v_cp != gv:
                distx = ar_cb[gnr_ind[gnr],gnr_ind[g_kld2_id[v_cp]]]
                cht_dists.append(distx)

        pos_non_inf = np.where(np.array(cht_dists) < math.inf)

        pct_non_inf = len(pos_non_inf[0])/len(cht)
        mean_non_inf = np.mean([cht_dists[i] for i in pos_non_inf[0]])

        cohrt_pcts_inf.append(1-pct_non_inf)
        cohrt_means_non_inf.append(mean_non_inf)

    # possible to weight by distance to parent of cohort, cohort size, both,
    # neither
    cohrt_pct_inf = np.mean(cohrt_pcts_inf)
    cohrt_mean_non_inf = np.mean(cohrt_means_non_inf)
    return(cohrt_pct_inf, cohrt_mean_non_inf)

def ar_cb_proc(gnr):
    """collects information about potential (penl) parents and children"""
    # would probably go easier but not really expensive
    
    t1 = time.time()
    ar_pos = gnr_ind[gnr]

    thd_res = []

    thds = [100, 0.1, 0.2, 0.3]
    thd_names = []


    for thd in thds:
        thd_prnts = ar_cb[ar_pos][np.where(ar_cb[ar_pos] < thd)]
        
        len_penl_prnts = len(thd_prnts)
        mean_penl_prnts = np.mean(thd_prnts)
        sum_penl_prnts = np.sum(thd_prnts)

        # children calculations
        thd_chirn = ar_cb[:,ar_pos][np.where(ar_cb[:,ar_pos] < thd)]

        len_penl_chirn = len(thd_chirn)
        mean_penl_chirn = np.mean(thd_chirn)
        sum_penl_chirn = np.sum(thd_chirn)

        thd_res.extend([len_penl_prnts, mean_penl_prnts, sum_penl_prnts,
                        len_penl_chirn, mean_penl_chirn, sum_penl_chirn])
        
        thd_names.extend(['len_penl_prnts_' + str(thd),
                          'mean_penl_prnts_' + str(thd),
                          'sum_penl_prnts_' + str(thd),
                          'len_penl_chirn_' + str(thd),
                          'mean_penl_chirn_' + str(thd),
                          'sum_penl_chirn_' + str(thd),])
    t2 = time.time()

    return(thd_res, thd_names)

# ** dfcx processing

def dfcx_proc(gnr):
    """generates a number of measures based on song data:
    - unq_artsts: number of unique artists in period
    - gnr_gini: see how unequally size (cnt * rel_weight) is distributed along artists
    - avg_age: average age of song, weighted by sz
    - age_sd: sd of age, weighted by sz
    - nbr_rlss_tprd: number of releases in genre in period
    - ttl_size: total size 
    - prop_rls_size: proportion of size of new new releases compared to total size
    - dist_mean: calculates euclidean distances for all songs (or a sample of 1k if more in gnr), avrg distance
    - dist_sd: sd of euclidean distances
    """

    dfcx = acst_gnr_dict[gnr]
    dfcx['sz'] = dfcx['cnt'] * dfcx['rel_weight']
    

    # artist variables: number, concentration of size
    unq_artsts = len(set(dfcx['artist']))
    # could even add concentration, like group playcount by artist, and then gini

    dfcx_grpd = dfcx[['sz', 'artist']].groupby('artist').sum()
    
    # dfcx_grpd.index[np.where(dfcx_grpd['sz'] > 100)[0]]
    gnr_gini = gini(np.array(dfcx_grpd['sz']))
    
    
    # age of songs:
    agex = t2_int - dfcx['erl_rls']
    avg_age, age_sd = weighted_avg_and_std(agex[np.where(agex > 0)[0]], weights = dfcx['sz'][np.where(agex > 0)[0]])
    

    # new releases: number, size, proportion to overall size
    rlss_tprd = dfcx[(dfcx['erl_rls'] > t1_int) & (dfcx['erl_rls'] < t2_int)]

    nbr_rlss_tprd = len(rlss_tprd)

    rlss_tprd_size = np.sum(rlss_tprd['sz'])
    ttl_size = np.sum(dfcx['sz'])

    prop_rls_size = rlss_tprd_size/ttl_size

    
    # average euclidean distance
    # sample for large genres
    if len(dfcx) > 1000:
        ids = sample(range(len(dfcx)), 1000)

        dfcx = dfcx.iloc[ids]

    distsx = euclidean_distances(dfcx[vrbls])

    dist_mean = np.mean(distsx[np.where(np.tril(distsx) > 0)])
    dist_sd = np.std(distsx[np.where(np.tril(distsx) > 0)])

    dfcx_names=['unq_artsts','gnr_gini','avg_age','age_sd','nbr_rlss_tprd','ttl_size','prop_rls_size',
                'dist_mean','dist_sd']
    dfcx_vlus = [unq_artsts, gnr_gini, avg_age, age_sd, nbr_rlss_tprd, ttl_size, prop_rls_size, dist_mean, dist_sd]

    return(dfcx_names, dfcx_vlus)


# * higher level  management functions

def gnr_t_prds(tdlt):
    time_start = datetime.date(datetime(2006,1,1))
    period_end = datetime.date(datetime(2012,12,20))

    # tdlt = 28

    time_periods = []

    while True:
        time_end = time_start + timedelta(days=tdlt)

        if time_end < period_end:
            time_range = [time_start, time_end]
            time_periods.append(time_range)

            time_start = time_start + timedelta(days=tdlt+1)
        else:
            break

    return(time_periods)




if __name__ == '__main__':
    time_periods = gnr_t_prds(28*3)

    res_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/'
    
    # tprd = time_periods[20]

    for tprd in time_periods:
        t1 = tprd[0].strftime('%Y-%m-%d')
        t2 = tprd[1].strftime('%Y-%m-%d')

        t1_dt = datetime.strptime(t1, '%Y-%m-%d')
        t2_dt = datetime.strptime(t2, '%Y-%m-%d')
        base_dt = datetime(1970, 1, 1)
        t1_int = (t1_dt - base_dt).days
        t2_int = (t2_dt - base_dt).days


        tp_id = time_periods.index(tprd)
        tp_clm = t1 + ' -- ' + t2

        print('set parameters')
        min_cnt = 10
        min_weight = 10
        min_rel_weight = 0.1
        min_tag_aprnc = 30
        d1 = t1
        d2 = t2
        

        vrbls=['dncblt','gender','timb_brt','tonal','voice','mood_acoustic',
               'mood_aggressive','mood_electronic','mood_happy','mood_party','mood_relaxed','mood_sad'] 

        print('construct dfc')

        dfc = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc, d1, d2, client, pd)

        gnrs = list(np.unique(dfc['tag']))

        print('construct acst gnr dict')

        acst_gnr_dict = dict_gnrgs(dfc, gnrs, pd)
        sz_dict, gnr_ind = gnrt_sup_dicts(acst_gnr_dict, gnrs)        

        print('construct acoustic edge list')
        # el_ttl = gnrt_acst_el(gnrs)
        el_ttl = gnrt_acst_el_mp(gnrs)
        sz_dict, gnr_ind = gnrt_sup_dicts(acst_gnr_dict,gnrs)


        print('construct acoustic graph')
        gac, w, w_std, w_std2, gac_id, vd, vdrv = gac_crubgs(el_ttl)

        print('construct acoustic mat')
        acst_mat = acst_arfy(el_ttl, vrbls, 3)

        t1 = time.time()
        print('construct kld mat')
        ar_cb = kld_mat_crubgs(gnrs)
        t2 = time.time()

        print('construct kld 3 parent edgelist')
        npr = 4
        kld2_el = kld_n_prnts(ar_cb, npr)

        print('construct kld graph')
        g_kld2, kld_sim, g_kld2_id, vd_kld2, vd_kld2_rv = kld_proc(kld2_el)

        print('extract features')
        # could be parallelized as well
        df_res = ftr_extrct()

        df_res['t1'] = t1
        df_res['t2'] = t2
        df_res['tp_id'] = tp_id
                
        print(df_res.shape)
        print('print to csv')
        df_res.to_csv(res_dir + tp_clm + '.csv')

        # raise Exception('done')

# * xprtl
# ** binarize weights

acst_mat_bn = np.zeros(acst_mat.shape)
acst_mat_bn[np.where(acst_mat > 0.1)] = 1

sums = np.sum(acst_mat_bn, axis=1)
nph(sums)

vrbl_nd_strs_raw = [[vrbl + str(i) for i in range(1,11)] for vrbl in vrbls]
vrbl_nd_strs = list(itertools.chain.from_iterable(vrbl_nd_strs_raw))

el_bin = []

for gnr in gnrs:
    gnr_ar_bin = acst_mat_bn[gnr_ind[gnr]]
    ftrs_bin = np.array(vrbl_nd_strs)[np.where(gnr_ar_bin == 1)]

    gnrs_el_bin = [(gnr, f) for f in ftrs_bin]
    el_bin = el_bin + gnrs_el_bin

g_bin = Graph()
g_bin_id = g_bin.add_edge_list(el_bin, hashed=True, string_vals=True)

vd_bin, vd_bin_rv = vd_fer(g_bin, g_bin_id)

cmps = all_cmps_crubgs(gnrs, vd_bin, 'product')

sims = vertex_similarity(g_bin, 'dice', vertex_pairs = cmps)

ovlp_ar = asym_sim(g_bin, gnrs, vd_bin)

# shows that high overlap doens't mean high similarity?
# high overlap doesn't mean high similarity because the similarity here is symmetric
# if there isn't much overlap for one genre, but super much for the other, for example i think

nph(sims_ar[np.where(ovlp_ar > 0.8)])
nph(ovlp_ar[np.where(ovlp_ar > 0.9)])

# subsetting with absolute stuff no good
bin_el1 = sbst_eler(ovlp_ar, operator.gt, 0.9)

g_hr_bin = Graph()
g_hr_bin_sim = g_hr_bin.new_edge_property('float')
g_hr_bin_id = g_hr_bin.add_edge_list(bin_el1, hashed = True, string_vals=True, eprops = [g_hr_bin_sim])

graph_pltr(g_hr_bin, g_hr_bin_id, 'acst_spc6.pdf', 1)

## not exactly sure if that works: should rewrite kld_n_prnts into general el function similar to sbst_eler
bin_el2 = kld_n_prnts(1-ovlp_ar ,4)


g_asym, asym_sim, g_asym_id, vd_asym, vd_asym_rv = kld_proc(bin_el2)
graph_pltr(g_asym, g_asym_id, 'acst_spc6.pdf', 1)


# should try with different thresholds (0.1, 0.15, 0.2) and see if difference



# ** song similiarity


    

# tx1 = time.time()
# dfcx_proc('electronic')
# tx2 = time.time()


# unq_artsts
# gnr_gini
# avg_age
# age_sd
# nbr_rlss_tprd
# ttl_size
# prop_rls_size
# dist_mean
# dist_sd



# metal genres seem to have skewed or even bimodal distributions
# but also i wonder if the other genres with normal distributions centered between 1 and 1.5 are ok
# basically means nothing is really similar to each other?
# 


# * scrap
## ** time durations

# ch_qry = 'SELECT time_d, count(time_d) FROM logs GROUP BY time_d'
# time_cnts = client.execute(ch_qry)
# time_pnts = [i[0] for i in time_cnts]
# cnts = [i[1] for i in time_cnts]

# ax = plt.axes()
# ax.plot(time_pnts, cnts)
# plt.show()

# qry = 'select time_d, uniq(usr) from logs group by time_d'
# time_cnts = client.execute(qry)
# time_pnts = [i[0] for i in time_cnts]
# unq_usrs = [i[1] for i in time_cnts]

# ax = plt.axes()
# ax.plot(time_pnts, unq_usrs)
# plt.show()


## ** speed up implementations


# is dict_gnrs (produces acst_gnr_dict) parallelizable?
# in principle yeah: can split pandas df, process rows separately,
# merging into one in the end is a bit work but not too much tbh
# is it worth it?
# takes 10 sec, 30sec for 6 months
# idk that's like 10% of the time (5 min), but kinda neglible against
# el_acst_mp: even with multiprocessing still 46 second -> 90 sec saved
# kld time: even when paralellized, still takes for fucking ever: 250 sec,
# tbh firefox took up a lot but still



## *** fucking done
# gnrt_acst_el is single core, can be parallelized tho, might be worth it
# kld mat also takes quite some time
# wonder if custom cython function would be faster
# seems to be already heavily using C funcs, so don't really think there's much to improve
