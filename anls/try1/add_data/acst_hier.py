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

from scipy.stats import entropy
from sklearn import preprocessing

from graph_tool.all import *
from graph_tool import *

from gnrl_funcs import get_dfs
from gnrl_funcs import dict_gnrgs

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

def gnrt_acst_el(acst_gnr_dict, gnrz):
    """generates edge list for acoustic space network"""
    
    # gnrz = acst_gnr_dict.keys()
    el_ttl = []
    sz_dict = {}
    gnr_ind = {}
    
    for gnr in gnrz:
    #     print(gnr)
        
        # dfc = get_df_cbmd(gnr)
        gnr_ind[gnr] = gnrz.index(gnr)
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
    return(el_ttl, sz_dict, gnr_ind)



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


def kld_mp(chnk):
    """multiprocessing function for KLD"""
    
    ents_ttl = []

    for gnr in chnk:

        i_id = gnr_ind[gnr]
        i_v = acst_mat[i_id]
        gnr_ents = []
        
        for k in gnrs:
            
            k_id = gnr_ind[k]
            k_v = acst_mat[k_id]
            
            b_zeros = np.where(k_v==0)
            a_sum_b_zeros = sum(i_v[b_zeros])
            prop_missing = a_sum_b_zeros/sum(i_v)
            
            if prop_missing == 0:
                ent = entropy(i_v, k_v)
                
            elif prop_missing < 0.05:
                
                i_v2 = np.delete(i_v, b_zeros)
                k_v2 = np.delete(k_v, b_zeros)

                ent = entropy(i_v2, k_v2)
            else:
                ent = math.inf

            gnr_ents.append(ent)
        ents_ttl.append(gnr_ents)
    return(ents_ttl)
                

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
    data = p.map(kld_mp, [i for i in gnr_chnks])
    t2=time.time()

    p.close()

    ar_lst = [np.array(i) for i in data]
    ar_cb = np.concatenate(ar_lst, axis=0)

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



# * feature extraction

## ** amount of musical space spanning
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

def all_cmps_crubgs(gnrs, vd, type):
    """constructs comparisons"""
    gnr_ids = [vd[i] for i in gnrs]
    lenx = len(gnrs)

    if type == "permutations":
        cmps = list(itertools.permutations(gnr_ids, 2))

    if type == "product":
        cmps = list(itertools.product(gnr_ids, repeat=2))
        
    return(cmps)



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
        res_dict[gnr]['prnt3_dvrg'] = prnt3_dvrg

        # from original data, might be interesting to weigh/add sds
        res_dict[gnr]['sz_raw'] = sz_dict[gnr]
        res_dict[gnr]['avg_weight_rel'] = np.mean(acst_gnr_dict[gnr]['rel_weight'])


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

    df_res = pd.DataFrame(res_dict).transpose()
    df_res['spngns_std'] = df_res['spngns']-min(df_res['spngns'])
    df_res['spngns_std'] = df_res['spngns_std']/max(df_res['spngns_std'])

    return(df_res)

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


# ** degree of space spanning
# use similar logic of omnivorousness



# for gnr in gnrs:

if __name__ == '__main__':
    # dates generation
    print('set parameters')
    min_cnt = 10
    min_weight = 10
    min_rel_weight = 0.1
    min_tag_aprnc = 30
    d1 = '2011-05-01'
    d2 = '2011-05-31'

    vrbls=['dncblt','gender','timb_brt','tonal','voice','mood_acoustic',
           'mood_aggressive','mood_electronic','mood_happy','mood_party','mood_relaxed','mood_sad'] 
    print('construct dfc')
    dfc = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc, d1, d2, client, pd)
    
    gnrs = list(np.unique(dfc['tag']))
    
    print('construct acst gnr dict')
    acst_gnr_dict = dict_gnrgs(dfc, gnrs, pd)

    print('construct acoustic edge list')
    el_ttl, sz_dict, gnr_ind = gnrt_acst_el(acst_gnr_dict, gnrs)

    print('construct acoustic graph')
    gac, w, w_std, w_std2, gac_id, vd, vdrv = gac_crubgs(el_ttl)

    print('construct acoustic mat')
    acst_mat = acst_arfy(el_ttl, vrbls, 3)

    print('construct kld mat')
    ar_cb = kld_mat_crubgs(gnrs)

    print('construct kld 3 parent edgelist')
    npr = 4
    kld2_el = kld_n_prnts(ar_cb, npr)

    print('construct kld graph')
    g_kld2, kld_sim, g_kld2_id, vd_kld2, vd_kld2_rv = kld_proc(kld2_el)

    print('extract features')
    df_res = ftr_extrct()
    print(df_res.shape)
    raise Exception('done')



