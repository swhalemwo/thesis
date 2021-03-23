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
from functools import partial
import argparse


import os
import operator
from datetime import datetime
from datetime import timedelta
from random import sample

from scipy.stats import entropy
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import Lasso
from sklearn.neighbors import KernelDensity

from graph_tool.all import *
from graph_tool import *

from gnrl_funcs import get_dfs
from gnrl_funcs import dict_gnrgs

# from gnrl_funcs import gini
# from gnrl_funcs import weighted_avg_and_std

from collections import Counter

client = Client(host='localhost', password='anudora', database='frrl')

# * auxilliary
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

def weighted_avg_and_std(np, values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


# * create acst network and all kinds of stuff



def asym_sim(gx, gnrs, vdx):
    """generates asym mat based on weights scaled by max, making more equally distributed genres more general"""
    cmps = all_cmps_crubgs(gnrs, vdx, 'product')

    all_sims = vertex_similarity(gx, 'dice', vertex_pairs = cmps)
    
    # don't think there's much sense in weights: i have weights because i standardize in np.hist
    # if i don't all proportionality constraints are of and larger ones will swallow small ones
    # but if i use weights the outdegree becomes the same which makes asymmetric similarity pointless
    # -> asymmetric similarity through overlap requires variation in out_degree
    # all_sims = vertex_similarity(gx, 'dice', vertex_pairs = cmps, eweight = w_std2)

    sims_rows = np.split(all_sims, len(gnrs))
    sims_ar = np.array(sims_rows)

    # deg_vec = [gx.vertex(vdx[i]).out_degree(weight=w_std2) for i in gnrs]
    deg_vec = [gx.vertex(vdx[i]).out_degree() for i in gnrs]

    # equivalent to adding the two outdegrees together 
    deg_ar = np.array([deg_vec]*len(gnrs))
    deg_ar2 = (deg_ar + np.array([deg_vec]*len(gnrs)).transpose())/2

    # see how much is actually in common, equivalent of multiplication with similarity
    cmn_ar = deg_ar2*sims_ar

    # see the percentage of what is in common for each genre
    ovlp_ar = cmn_ar/deg_ar
    return(ovlp_ar)

# gx = g_trks
# vdx = g_trks_vd
# sims_ar = gnr_sim_ar

def asym_sim2(gx, vdx, gnrs, sims_ar, wts):
    """calculates asymmetric similarities from graph, weights similarity mat (also needs vd and gnrs)"""
    
    deg_vec = [gx.vertex(vdx[i]).in_degree(wts) for i in gnrs]

    # equivalent to adding the two outdegrees together 
    deg_ar = np.array([deg_vec]*len(gnrs))
    deg_ar2 = (deg_ar + np.array([deg_vec]*len(gnrs)).transpose())/2

    # see how much is actually in common, equivalent of multiplication with similarity
    cmn_ar = deg_ar2*sims_ar

    # see the percentage of what is in common for each genre
    ovlp_ar = cmn_ar/deg_ar
    return(ovlp_ar)


def gnrt_acst_el(acst_gnr_dict, nbr_cls, gnrs):
    """generates edge list for acoustic space network"""
    
    el_ttl = []
    
    for gnr in gnrs:
        dfcx = acst_gnr_dict[gnr]

        gnr_cnt = len(dfcx)
        
        el_gnr = []
        # print(len(el_gnr))

        wts = dfcx['rel_weight'] * dfcx['cnt']

        for vrbl in vrbls:
            if nbr_cls == 1:
                
                vlu = np.average(dfcx[vrbl], weights = wts)
                nds1 = vrbl + '1'
                nds2 = gnr

                elx = [[nds2, nds1, sum(wts), vlu, vlu]]

            else:
                bins = np.arange(0, 1 + 1/nbr_cls, 1/nbr_cls)
                a1, a0 = np.histogram(dfcx[vrbl], bins=nbr_cls, weights=wts)
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


def gnrt_acst_el_mp(gnrs, acst_gnr_dict, nbr_cls):
    """parallelizes the acoustic edgelist generation process"""

    # iterable = [1, 2, 3, 4, 5]
    # pool = multiprocessing.Pool()
    # a = "hi"
    # b = "there"
    # func = partial(f, a, b)
    # pool.map(func, iterable)
    # pool.close()
    # pool.join()

    NO_CHUNKS = 3
    gnr_chnks = list(split(gnrs, NO_CHUNKS))

    func = partial(gnrt_acst_el, acst_gnr_dict, nbr_cls)

    t1 = time.time()
    p = Pool(processes=NO_CHUNKS)
    data = p.map(func, [i for i in gnr_chnks])
    t2=time.time()

    p.close()
    p.join()

    el_ttl = list(itertools.chain.from_iterable(data))
    return(el_ttl)



def gnrt_sup_dicts(acst_gnr_dict,gnrs):
    """generate supplementary dicts: size, genre index, weight of genre?, volume (count times rel. weight)"""
    sz_dict = {}
    gnr_ind = {}
    waet_dict = {}
    vol_dict = {}

    for gnr in gnrs:
        sz_dict[gnr] = len(acst_gnr_dict[gnr])
        gnr_ind[gnr] = gnrs.index(gnr)
        waet_dict[gnr] = np.mean(acst_gnr_dict[gnr]['rel_weight'])
        vol_dict[gnr] = sum(np.array(acst_gnr_dict[gnr]['cnt']) * np.array(acst_gnr_dict[gnr]['rel_weight']))
        
    return(sz_dict, gnr_ind, waet_dict, vol_dict)

# sz_dict, gnr_ind = gnrt_sup_dicts(acst_gnr_dict, gnrs)



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



def acst_arfy(el_ttl, vrbls, el_pos, gnrs, nbr_cls):
    """converts edge list to matrix"""

    acst_mat = np.empty([len(gnrs), len(vrbls)*nbr_cls])

    c = ypos =xpos = 0

    for i in el_ttl:
        itempos = [ypos, xpos]
        # have to make sure to select right one here
        vlu = i[el_pos]

        acst_mat[ypos, xpos] = vlu

        xpos+=1

        if xpos == len(vrbls)*nbr_cls:
            xpos = 0
            ypos+=1

        c+=1
    return(acst_mat)

def krnl_acst(acst_gnr_dict, nbr_cls, vrbls, gnrs):
    """generates acst kernel (gaussian smoothing) with bandwith of half of cell size"""
    
    bw = 1/(2*nbr_cls)
    

    x_vlus = np.array(np.linspace(0, 1, nbr_cls))[:, np.newaxis]
    # should i start x_vlus at 0 or 0 + bw? 
    # x_vlus = np.array(np.linspace(0 + bw, 1 -bw, nbr_cls))[:, np.newaxis]
    
    chnk_scrs = []

    for gnr in gnrs:
        dfcx = acst_gnr_dict[gnr]
        wts = dfcx['rel_weight'] * dfcx['cnt']
        
        gnr_scrs = []

        for vrbl in vrbls:
            
            krnl = KernelDensity(bandwidth = bw).fit(np.array(dfcx[[vrbl]]), sample_weight = wts)
            scrs = krnl.score_samples(x_vlus)
            scrs_exp = np.exp(scrs)
            scrs_nml = scrs_exp/sum(scrs_exp)
            
            gnr_scrs.append(scrs_nml)
    
        gnr_scrs2 = np.concatenate(gnr_scrs)
        chnk_scrs.append(gnr_scrs2)
        
    ttl_scrs = np.concatenate([chnk_scrs], axis = 1)
    
    return(ttl_scrs)

def krnl_acst_mp(gnrs, acst_gnr_dict, nbr_cls, vrbls):
    """multiproccessing managing function for acst kernel"""
    NO_CHUNKS = 6
    gnr_chnks = list(split(gnrs, NO_CHUNKS))

    func = partial(krnl_acst, acst_gnr_dict, nbr_cls, vrbls)
    
    t1 = time.time()
    p = Pool(processes=NO_CHUNKS)
    data = p.map(func, [i for i in gnr_chnks])
    t2=time.time()

    p.close()
    p.join()

    acst_mat = np.concatenate(data, axis = 0)
    return(acst_mat)

# acst_mat = acst_arfy(el_ttl, vrbls,3)


# ** KLD
# needs functionizing and parallel processing 

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))



def kld_mp2(acst_mat, gnrs, chnk):
    """ improved kdl with https://www.oipapio.com/question-4293090, no more inf fixing but shouldn't be needed it it's proper subsets, """
    # comparison with previous shows that places which are now also infinite are generally super high
    
    start_col = gnrs.index(chnk[0])
    end_col = gnrs.index(chnk[-1])+1
    
    # kldx = entropy(acst_mat[start_col:end_col,:].T[:,:,None], acst_mat.T[:,None,:])
    
    # acst_mat[start_col:end_col,:].T[:,:,None]: shape: (84, 10, 1)
    # https://stackoverflow.com/questions/34007028/efficient-way-of-computing-kullback-leibler-divergence-in-python
    # kldx = rel_entr(acst_mat[start_col:end_col,:].T[:,:,None], acst_mat.T[:,None,:]).sum(axis=0)
    
    # kldx = entropy_custom(acst_mat[start_col:end_col,:].T[:,:,None], acst_mat.T[:,None,:])
    
    # a,b = np.broadcast(acst_mat.T[start_col:end_col,:], acst_mat.T[:,None,:])
    
    # a = np.broadcast(acst_mat[start_col:end_col,:].T[:,:,None], acst_mat.T[:,None,:])
    # kldx = entroy(
    
    # a, b = np.broadcast_arrays(distributions.T[:,:,None], distributions.T[:,None,:])
    # pairwise_klds1 = entropy(a, b)
    
    # x= np.broadcast(acst_mat[start_col:end_col,:].T[:,:,None], acst_mat.T[:,None,:])
    
    kldx = entropy_custom(acst_mat[start_col:end_col,:].T[:,:,None], acst_mat.T[:,None,:])
    
    return(kldx)


    

def entropy_custom(pk, qk=None, base=None, axis=0):
    """custom version of entropy without shape requirements to enable broadcasting"""
    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        vec = entr(pk)
    else:
        qk = np.asarray(qk)
        # if qk.shape != pk.shape:
        #     raise ValueError("qk and pk must have same shape.")
        qk = 1.0*qk / np.sum(qk, axis=axis, keepdims=True)
        vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= log(base)
    return S




# t1 = time.time()
# for i in range(1000):
    # entropy(list(k_v)*10, list(i_v)*10)
    # entropy(list(k_v), list(i_v))
# t2 = time.time()
# increasing number of elements only slowly increases time (10*elements: twice as slow): logarithmic? 

def kld_mat_crubgs(gnrs, acst_mat):
    """multiproccesses acst_mat to create kld mat"""
    NO_CHUNKS = 3
    gnr_chnks = list(split(gnrs, NO_CHUNKS))

    func2 = partial(kld_mp2, acst_mat, gnrs)

    p = Pool(processes=NO_CHUNKS)

    t1=time.time()
    data = p.map(func2, [i for i in gnr_chnks])
    t2=time.time()

    # t1=time.time()
    # data2 = p.map(kld_mp, [i for i in gnr_chnks])
    # t2=time.time()

    p.close()
    p.join()

    # ar_lst = [np.array(i) for i in data2]
    # ar_cb = np.concatenate(ar_lst, axis=0)

    ar_cb = np.concatenate(data, axis=0)

    return(ar_cb)

# need to deal with inf values
# where do they come from? 0s in A i think
# assess badness of fit: sum of cells of a that are 0 in b should not be above X (0.95)

# * integrate diagnotisticy to now schema model

def kld_schema(acst_mat, nbr_cls, vrbl_set):
    """calculate KLD for each vrbl separately and weight by diagnosticity"""
    
    vrbl_vec = list(itertools.chain.from_iterable([[i] * nbr_cls for i in vrbls]))
    
    res_mat = []
    
    for vrbl in vrbl_set:
        
        rel_cols = np.where(np.array(vrbl_vec) == vrbl)[0]
        vrbl_mat = acst_mat[:,rel_cols]

        vrbl_maxs = np.max(vrbl_mat, axis = 1)
        vrbl_maxs_mat = np.array([vrbl_maxs] * nbr_cls).T

        vrbl_ttl = vrbl_mat/vrbl_maxs_mat
        vrbl_diagnst = 1/np.sum(vrbl_ttl, axis = 1)
        
        diag_mat = np.array([vrbl_diagnst] * len(vrbl_diagnst)).T
        
        # diag_mat2 = diag_mat.T
        # diag_mat_pruc = diag_mat * diag_mat2

        # which diag_array do i need? 
        # would say weights shoyld be in rows..
        
        kldx = entropy(vrbl_mat.T[:,:,None], vrbl_mat.T[:,None,:])

        kldx_wtd = kldx * diag_mat
        res_mat.append(kldx_wtd)

    res_mat_ttl = sum(res_mat)
    return(res_mat_ttl)


# g1 = gnrs[10]
# g2 = gnrs[20]

# entropy(vrbl_mat[10], vrbl_mat[20])

# sum([i[0] * np.log(i[0]/i[1]) for i in zip(vrbl_mat[10], vrbl_mat[20])])
# kld is divergence of p from q
# q can't be 0
# importance of p matters more me thinks
# have to take reverse of vector mat: 
# rows in ar_cb show the distances of the row from the potential parents





def kld_schema_mp(vrbls, acst_mat, nbr_cls):

    NO_CHUNKS = 3
    vrbl_chnks = list(split(vrbls, NO_CHUNKS))

    func = partial(kld_schema, acst_mat, nbr_cls)

    t1 = time.time()
    p = Pool(processes=NO_CHUNKS)
    data = p.map(func, [i for i in vrbl_chnks])
    t2=time.time()

    p.close()
    p.join()

    kld_schema_mat = sum(data)
    return(kld_schema_mat)





# *** get N closest parents

def kld_n_prnts(ar_cb, npr, gnrs, gnr_ind):
    """generates edgelist by taking npr (number of parents) lowest values of row of asym kld mat"""

    npr = npr + 1
    
    kld2_el = []

    for i in gnrs:
        i_id = gnr_ind[i]
        sims = ar_cb[i_id]

        idx = np.argpartition(sims, npr)
        prnts = idx[0:npr]
        vlus = sims[idx[0:npr]]

        for k in zip(prnts, vlus):
            if k[0] == i_id:
                pass
            else:
                kld2_el.append((gnrs[k[0]], gnrs[i_id], k[1]))
                
    return(kld2_el)

# oprtr = operator.lt

def asym_n_prnts(asym_ar, oprtr, npr):
    elx = []

    for i in gnrs:
        i_id = gnr_ind[i]
        vctr = asym_ar[i_id]

        



def kld_proc(kld2_el):
    
    g_kld2 = Graph()
    
    g_kld2.ep['kld_sim'] = g_kld2.new_edge_property('double')
    g_kld2.vp['id'] = g_kld2.add_edge_list(kld2_el, hashed=True, eprops = [g_kld2.ep.kld_sim])

    vd_kld2,vd_kld2_rv = vd_fer(g_kld2, g_kld2.vp.id)

    # get int e_vp for plotting
    # kld_sim_int = g_kld2.new_edge_property('int16_t')
    # for e in g_kld2.edges():
    #     kld_sim_int[e] = math.ceil(kld_sim[e]*100)

    # graph_pltr(g_kld2, g_kld2_id, 'acst_spc5.pdf', kld_sim_int)

    return(g_kld2, vd_kld2, vd_kld2_rv)


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

def ftr_extrct(gnrs, nbr_cls, gnr_ind, ar_cb, g_kld2, vd_kld2, 
               acst_gnr_dict, sz_dict, vol_dict, acst_mat):
               # ar_cb2, g_kld3, vd_kld3):

    NO_CHUNKS = 3
    chnks = list(split(gnrs, NO_CHUNKS))
    print(chnks)

    # print('prep spanning')
    # cmps_rel, sim_v = gnr_span_prep(gac, vrbls, nbr_cls, vd, w)
    # print('prep spanning done')


    func3 = partial(ftr_extrct_mp, nbr_cls, gnr_ind, 
                    ar_cb, g_kld2, vd_kld2, acst_gnr_dict, 
                    sz_dict, vol_dict, acst_mat)
                    # , cmps_rel, sim_v) gac, vd, w,
                    # ar_cb2, g_kld3, vd_kld3)
    
    p = Pool(processes=NO_CHUNKS)
    res_data = p.map(func3, [i for i in chnks])
    p.close()
    p.join()

    super_dict = {}
    for d in res_data:
        for k, v in d.items(): 
            super_dict[k] = v

    df_res = pd.DataFrame(super_dict).transpose()

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df_res)
    
    # df_res['spngns_std'] = df_res['spngns']-min(df_res['spngns'])
    # df_res['spngns_std'] = df_res['spngns_std']/max(df_res['spngns_std'])

    return(df_res)


def hier_ftrs(gnr, g_hier, vd_hier, ar_cbx, acst_mat, vol_dict, gnr_ind, hst_lbl):
    """processes hierarchical sets (hsts)"""
    prnt_stats_names, prnt_stats_vlus = prnt_stats(gnr, gnr_ind, ar_cbx, g_hier, vd_hier, acst_mat, vol_dict)
    
    cohrt_vlu_names, cohrt_vlus = chrt_proc(gnr, g_hier, gnr_ind, vd_hier, ar_cbx, acst_mat, vol_dict)

    all_names = prnt_stats_names + cohrt_vlu_names
    all_names2 = [hst_lbl + '_' + i for i in all_names]
    
    all_vlus = prnt_stats_vlus + cohrt_vlus

    return(all_names2, all_vlus)


def ftr_extrct_mp(nbr_cls, gnr_ind, ar_cb, g_kld2, vd_kld2, 
                  acst_gnr_dict, sz_dict, vol_dict, acst_mat, gnrs):
                  # ar_cb2, g_kld3, vd_kld3, ):
    # gac, vd, w, w_std, 
    # , cmps_rel, sim_v, gnrs)
    # seems to sometimes require self, somtimes not??
    """extracts features like a boss"""

    # print('asdf')

    res_dict = {}
    for gnr in gnrs:
        res_dict[gnr] = {}


    print('start iterating')
    for gnr in gnrs:
        # generate a whole bunch of measures

        hier_sets = {'smpl_ftrs': [g_kld2, vd_kld2, ar_cb]}
                    # 'cell_cmbs': [g_kld3, vd_kld3, ar_cb2]}

        for hst in hier_sets.keys():
            if gnr in hier_sets[hst][1]:
                hstv = hier_sets[hst]
                if len(list(hstv[0].vertex(hstv[1][gnr]).in_neighbors())) > 0:
                    hier_names, hier_vlus = hier_ftrs(gnr, hstv[0], hstv[1], hstv[2], acst_mat, vol_dict, gnr_ind, hst)
                    for i in zip(hier_names, hier_vlus):
                        res_dict[gnr][i[0]] = i[1]

        # prnt_stats_names, prnt_stats_vlus = prnt_stats(gnr, gnr_ind, ar_cb, g_kld2, vd_kld2, acst_mat, vol_dict)
        # for i in zip(prnt_stats_names, prnt_stats_vlus):
        #     res_dict[gnr][i[0]] = i[1]

        # # cohorts
        # cohrt_vlu_names, cohrt_vlus = chrt_proc(gnr, g_kld2, gnr_ind, vd_kld2, ar_cb, acst_mat, vol_dict)
        # for i in zip(cohrt_vlu_names, cohrt_vlus):
        #     res_dict[gnr][i[0]] = i[1]

        thd_res, thd_names = ar_cb_proc(gnr, gnr_ind, ar_cb)
        for i in zip(thd_names, thd_res):
            res_dict[gnr][i[0]] = i[1]

        # spanningness
        # spngns = gnr_mus_spc_spng(gnr, cmps_rel, sim_v, gac, vd, w_std)
        # res_dict[gnr]['spngns'] = spngns
        
        tx5 = time.time()
        
        # CHECK IF ARGUMETNS OF SPECIFIC EXTRACTION FUNCTIONS ARE ALL IN MAIN EXTRACTION FUNCTION

        # dfcx stuff
        # for gnr in gnrs:
        
        print(gnr)
        dfcx_names, dfcx_vlus = dfcx_proc(gnr, acst_gnr_dict, vrbls, d2_int)
        for i in zip(dfcx_names, dfcx_vlus):
            res_dict[gnr][i[0]] = i[1]

        tx6 = time.time()
        # add to dfcx_proc
        # from original data, might be interesting to weigh/add sds
        res_dict[gnr]['sz_raw'] = sz_dict[gnr]
        res_dict[gnr]['volm'] = vol_dict[gnr]
        res_dict[gnr]['avg_weight_rel'] = np.mean(acst_gnr_dict[gnr]['rel_weight'])
        res_dict[gnr]['avg_weight_rel_wtd'] = np.average(acst_gnr_dict[gnr]['rel_weight'], weights = acst_gnr_dict[gnr]['cnt'])

        tx7 = time.time()
    
    return(res_dict)

# ** prnt stats
def prnt_stats(gnr, gnr_ind, ar_cb, g_kld2, vd_kld2, acst_mat, vol_dict):
    """calculates parent stats: 
    prnt3_dvrgs: sum of divergences from parents, 
    clst_prnt: distance to closest parent, 
    mean_prnt_dvrg: how far prnts are apart, 
    prnt_odg, prnt_odg_wtd: out-degree of parents (unweighted and weighted by distance)
    prnt_plcnt: playcount of parents, with sd
    """

    gv = g_kld2.vertex(vd_kld2[gnr])

    prnt3_dvrg = gv.in_degree(g_kld2.ep.kld_sim)
    clst_prnt = min([g_kld2.ep.kld_sim[v] for v in gv.in_edges()])

    prnts = [g_kld2.vp.id[i] for i in gv.in_neighbors()]
    prnt_vs = [i for i in gv.in_neighbors()]
    prnt_ids = [gnr_ind[i] for i in prnts]

    prnt_plcnt = np.sum([vol_dict[i] for i in prnts])
    prnt_plcnt_sd = np.std([vol_dict[i] for i in prnts])

    prnt_cmps = list(itertools.permutations(prnt_ids,2))
    prnt_cmps_names = list(itertools.permutations(prnts,2))

    prnt_sims_mat = cosine_similarity(acst_mat[prnt_ids])
    prnt_sims = prnt_sims_mat[np.where(np.triu(prnt_sims_mat, k=1) > 0,)]
    
    mean_prnt_sim = np.mean(prnt_sims)

    # can also use KLD now

    prnt_klds = [ar_cb[i] for i in prnt_cmps]
    mean_prnt_kld = np.mean(prnt_klds)
    
    prnt_cmps_wts = [vol_dict[i[0]] + vol_dict[i[1]] for i in prnt_cmps_names]
    mean_prnt_kld_wtd = np.average(prnt_klds, weights = prnt_cmps_wts)
    
    # may have to divide by 1 (or other thing to get distance), not quite clear now
    
    prnt_odg = np.mean([prnt_v.out_degree() for prnt_v in prnt_vs])
    prnt_odg_wtd = np.mean([prnt_v.out_degree() * g_kld2.ep.kld_sim[g_kld2.edge(prnt_v,gv)] for prnt_v in prnt_vs])

    prnt_stats_names = ['prnt3_dvrg', 'clst_prnt', 'mean_prnt_sim', 'prnt_odg', 
                        'prnt_odg_wtd', 'prnt_plcnt', 'prnt_plcnt_sd', 'mean_prnt_kld', 'mean_prnt_kld_wtd']
    prnt_stats_vlus = [prnt3_dvrg, clst_prnt, mean_prnt_sim, prnt_odg, 
                       prnt_odg_wtd, prnt_plcnt, prnt_plcnt_sd, mean_prnt_kld, mean_prnt_kld_wtd]

    return(prnt_stats_names, prnt_stats_vlus)




# average of similarities of indegrees
# superordinates

# spr_ord = list(g_kld2.vertex(vd_kld2[gnr]).in_neighbors())

# sim_vlu[ghrac.edge(gv, v)]
# sim_vlu[ghrac.edge(v, gv)]


# ** cohort processing
def chrt_proc(gnr, g_kld2, gnr_ind, vd_kld2, ar_cb, acst_mat, vol_dict):
    """generates all kinds of measures related to cohorts
    - cohrt_pct_inf: mean (of each cohort) of proportion of cohort members to which KLD is infinite
    - cohrt_mean_non_inf: average mean of the KLDs that are not infinite
    - cohrt_mean_non_inf_wtd: weighted average mean of the KLDs that are not infinite
    - cohrt_mean_cos_dists_wtd: volume-weighted mean of acst_mat cosine distance between genre and cohort members
    - cohrt_mean_cos_dists_uwtd: unweighted mean of acst_mat cosine distance between genre and cohort members
    - cohrt_len: number of genres in all cohorts
    - cohrt_vol_sum: sum of cohorts members volume
    - cohrt_vol_mean: mean of cohort members volume
    - cohrt_vol_sd: sd of cohort members volume
    """
    
    gv = g_kld2.vertex(vd_kld2[gnr])
    prnts = list(gv.in_neighbors())
    cohrts = [list(pr.out_neighbors()) for pr in prnts]
    cohrt_mbrs_dupl = list(itertools.chain.from_iterable(cohrts))
    cohrt_mbrs = list(np.unique(cohrt_mbrs_dupl))
    
    # cohrt_ids = [[gnr_ind[g_kld2.vp.id[i]] for i in c] for c in cohrts]
    
    # why should weight matter for cosine distance?
    # because if cohort is dominated by one genre, that's what matters
    # just do both duh
            
    # think should distinguish between total mean of all cohort members and mean of cohorts
    # not sure if really different but whatever
    
    # for cht in cohrts:
    #     if len(cht) > 1:
    cht_klds = []
    cht_sizes = []
    cht_cos_dists =  []

    for v_cp in cohrt_mbrs:
        kldx = ar_cb[gnr_ind[gnr],gnr_ind[g_kld2.vp.id[v_cp]]]
        cht_klds.append(kldx)

        cos_dist = cosine(acst_mat[gnr_ind[gnr]], acst_mat[gnr_ind[g_kld2.vp.id[v_cp]]])
        cht_cos_dists.append(cos_dist)

        cht_sizes.append(vol_dict[g_kld2.vp.id[v_cp]])

    pos_non_inf = np.where(np.array(cht_klds) < math.inf)
    pct_non_inf = len(pos_non_inf[0])/len(cht_klds)
    

    cohrt_mean_cos_dists_wtd, sd_cos_dist_wtd  = weighted_avg_and_std(np, cht_cos_dists, cht_sizes)
    cohrt_mean_cos_dists_uwtd = np.mean(cht_cos_dists)

    cohrt_mean_non_inf = np.mean([cht_klds[i] for i in pos_non_inf[0]])
    cohrt_mean_non_inf_wtd = np.average([cht_klds[i] for i in pos_non_inf[0]], weights=cht_sizes)

    cohrt_len = len(cht_sizes)
    cohrt_vol_sum = sum(cht_sizes)
    cohrt_vol_mean = np.mean(cht_sizes)
    cohrt_vol_sd = np.std(cht_sizes)

    cohrt_med = np.median(cht_sizes)

    cohrt_ovlp = 1-(len(cohrt_mbrs)/len(cohrt_mbrs_dupl))

    cohrt_vlu_names = ['pct_non_inf', 'cohrt_mean_non_inf', 'cohrt_mean_non_inf_wtd', 'cohrt_mean_cos_dists_wtd',
                       'cohrt_mean_cos_dists_uwtd', 'cohrt_len',
                       'cohrt_vol_sum', 'cohrt_vol_mean', 'cohrt_vol_sd', 'cohrt_ovlp', 'cohrt_med']

    cohrt_vlus = [pct_non_inf, cohrt_mean_non_inf, cohrt_mean_non_inf_wtd, cohrt_mean_cos_dists_wtd,
                  cohrt_mean_cos_dists_uwtd, cohrt_len,
                  cohrt_vol_sum, cohrt_vol_mean, cohrt_vol_sd, cohrt_ovlp, cohrt_med]

    return(cohrt_vlu_names, cohrt_vlus)

# how to handle case where all parents have only gnr as child
# set stuff to 0? 


# ** ar_cb processing
def ar_cb_proc(gnr, gnr_ind, ar_cb):
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

def dfcx_proc(gnr, acst_gnr_dict, vrbls, d2_int):
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

    # speed improvements:
    # - between tx7 and tx6
    # - between tx3 and tx2
    # cant really improve either much
    
    dfcx = acst_gnr_dict[gnr]
    dfcx['sz'] = dfcx['cnt'] * dfcx['rel_weight']
    
    # artist variables: number, concentration of size
    unq_artsts = len(set(dfcx['artist']))
    # could even add concentration, like group playcount by artist, and then gini

    dfcx_grpd = dfcx[['sz', 'artist']].groupby('artist').sum()
    
    # dfcx_grpd.index[np.where(dfcx_grpd['sz'] > 100)[0]]
    gnr_gini = gini(np.array(dfcx_grpd['sz']))
    
    # age of songs:
    agex = d2_int - dfcx['erl_rls']
    avg_age, age_sd = weighted_avg_and_std(np, agex[np.where(agex > 0)[0]], weights = dfcx['sz'][np.where(agex > 0)[0]])
    
    # new releases: number, size, proportion to overall size
    rlss_tprd = dfcx[(dfcx['erl_rls'] > d1_int) & (dfcx['erl_rls'] < d2_int)]

    nbr_rlss_tprd = len(rlss_tprd)
    
    rlss_tprd_size = np.sum(rlss_tprd['sz'])
    ttl_size = np.sum(dfcx['sz'])

    prop_rls_size = rlss_tprd_size/ttl_size
    
    # average euclidean distance
    # sample for large genres
    if len(dfcx) > 1000:
        ids = sample(range(len(dfcx)), 750)

        dfcx = dfcx.iloc[ids]

    size_mat = np.array([dfcx['sz']]*len(dfcx))
    size_mat2 = size_mat + size_mat.T

    dists_euc = euclidean_distances(dfcx[vrbls])

    dist_euc_mean_uwtd = np.mean(dists_euc[np.where(np.tril(dists_euc) > 0)])
    dist_euc_mean_wtd = np.average(dists_euc[np.where(np.tril(dists_euc) > 0)],
                                   weights = size_mat2[np.where(np.tril(dists_euc) > 0)])
    dist_euc_sd = np.std(dists_euc[np.where(np.tril(dists_euc) > 0)])
    
    cos_sims = cosine_similarity(dfcx[vrbls])

    cos_sims_mean_uwtd = np.mean(cos_sims[np.where(np.tril(cos_sims) > 0)])
    cos_sims_mean_wtd = np.average(cos_sims[np.where(np.tril(cos_sims) > 0)],
                               weights = size_mat2[np.where(np.tril(cos_sims) > 0)])

    dfcx_names=['unq_artsts','gnr_gini','avg_age','age_sd','nbr_rlss_tprd','ttl_size','prop_rls_size',
                'dist_euc_mean_uwtd', 'dist_euc_mean_wtd', 'dist_euc_sd', 'cos_sims_mean_uwtd', 'cos_sims_mean_wtd']
    dfcx_vlus = [unq_artsts, gnr_gini, avg_age, age_sd, nbr_rlss_tprd, ttl_size, prop_rls_size,
                 dist_euc_mean_uwtd, dist_euc_mean_wtd, dist_euc_sd, cos_sims_mean_uwtd, cos_sims_mean_wtd]

    return(dfcx_names, dfcx_vlus)


# * higher level  management functions

def gnr_t_prds(tdlt):
    # time_start = datetime.date(datetime(2006,1,1))
    time_start = datetime.date(datetime(2008,9,1))

    # period_end = datetime.date(datetime(2012,12,20))
    period_end = datetime.date(datetime(2012,9,30))

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

def ptn_proc(ptn):
    """generates g_kld and df_res for a partition"""
    # do i need other things too?
    # acst_mat: assess degree of acoustic similarity
    # gnr_ind: process stuffz
    
    print('construct dfc')
    dfc = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc,
                  min_unq_artsts, max_propx1, max_propx2, d1, d2, ptn,
                  usr_dedcgs, tag_plcnt, unq_usrs,
                  client, pd)

    gnrs = list(np.unique(dfc['tag']))
    artsts = list(np.unique(dfc['artist']))
    trks = list(np.unique(dfc['lfm_id']))

    print('construct acst gnr dict')

    nbr_cls = 7
    acst_gnr_dict = dict_gnrgs(dfc, gnrs, pd)
    sz_dict, gnr_ind, waet_dict, vol_dict = gnrt_sup_dicts(acst_gnr_dict, gnrs)

    # NEED TO CHECK IF GENRES ARE ALREADY ESTABLISHED, OTHERWISE DELETE SMALL ONES
    # WRITE FIRST all that are, then pop those that aren't
    # save dfc? might allow it to filter faster afterwards
    # is large tho
    # dfc.to_csv(res_dir + 'dfc_' + tp_clm + '.csv')
    
    
    debug_file = "debug.csv"
    
    # genre counts as established if size >= 20
    # once established, keeps on counting even if atm not fulfilling the criteria
    est_gnrs = [[i, tp_id] for i in gnrs if sz_dict[i] >= min_inst_cnt]
    
    with open(res_dir + debug_file, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(est_gnrs)
        
    with open(res_dir + debug_file, 'r') as fi:
        rdr = csv.reader(fi)
        est_gnrs2 = [r[0] for r in rdr]
    
    non_staeb_gnrs = list(set(gnrs) - set(est_gnrs2))
    
    gnrs = [i for i in gnrs if i in set(est_gnrs2)]
    
    sz_dict, gnr_ind, waet_dict, vol_dict = gnrt_sup_dicts(acst_gnr_dict, gnrs)
    
    # acst_gnr_dict_bu = acst_gnr_dict
    # i think i don't even have to really delete; just reset the info dicts
    # for gf in non_staeb_gnrs:
    #     acst_gnr_dict.pop(gf)
    #     sz_dict.pop
    print('construct acst mat')

    acst_mat = krnl_acst_mp(gnrs, acst_gnr_dict, nbr_cls)


    print('construct kld mat')
    # ar_cb = kld_schema_mp(vrbls, acst_mat, nbr_cls)
    ar_cb = kld_mat_crubgs(gnrs, acst_mat)


    print('construct kld 3 parent edgelist')
    # could loop over npr 1-5, add prnt to column names? 
    npr = 2
    kld2_el = kld_n_prnts(ar_cb, npr, gnrs, gnr_ind)

    print('construct kld graph')
    g_kld2, vd_kld2, vd_kld2_rv = kld_proc(kld2_el)
    # graph_pltr(g_kld2, g_kld2.vp.id, res_dir +'/5_cell_space.pdf', 1.0)

    
    print('extract features')

    tx1 = time.time()
        

    df_res = ftr_extrct(gnrs, nbr_cls, gnr_ind, ar_cb, 
                        g_kld2, vd_kld2 , acst_gnr_dict, sz_dict, vol_dict, acst_mat)
                        # ar_cb2, g_kld3, vd_kld3)
    tx2 = time.time()

    df_res['entrp'] = entropy(acst_mat.T)
    
    ret_dict = {'g_kld2':g_kld2, 'df_res':df_res, 'gnr_ind':gnr_ind, 'acst_mat':acst_mat}

    return(ret_dict)

def mult_dice(ll_of_lls):
    """calculates all pairwise dice similarities for a list of lists"""
    x_sims = []
    c =0
    
    if len(ll_of_lls) ==1:
        return(1)
    
    for i1 in range(len(ll_of_lls)):
        for i2 in range(c, len(ll_of_lls)):
            if i1 == i2:
                pass
            else:
                # print(i1,i2)
                s1 = set(ll_of_lls[i1])
                s2 = set(ll_of_lls[i2])
                if len(s1) > 0 and len(s2) > 0:
                    sim = (2*len(s1.intersection(s2)))/(len(s1)+len(s2))
                else:
                    sim = 0
                x_sims.append(sim)
        c+=1
    
    return(x_sims)


def ptn_eval(ptns, ptn_obj_dict):

    all_gnrs = [ptn_obj_dict[i]['gnr_ind'].keys() for i in ptns]
    all_gnrs2 = itertools.chain.from_iterable(all_gnrs)
    all_gnrs3 = set(all_gnrs2)

    [print(len(ptn_obj_dict[i]['gnr_ind'])) for i in ptns]

    gnr_proc_dict= {}

    for gnr in all_gnrs3:
        print(gnr)

        rel_ptns = []
        for ptn in ptns:
            if gnr in ptn_obj_dict[ptn]['gnr_ind'].keys():
                rel_ptns.append(ptn)

        nbr_cls = 5
        gnr_acsts = np.zeros((nbr_cls*len(vrbls),len(rel_ptns)))

        gnr_prnts = []
        gnr_chirn = []

        ord_var_dict = {}
        ord_vars = ptn_obj_dict[0]['df_res'].columns
        for i in ord_vars:
            ord_var_dict[i] = []

        ptn_waets = []

        # maybe first get all relevant partions? 
        for ptn in rel_ptns:
            # get weights
            
            ptn_waets.append(ptn_obj_dict[ptn]['df_res'].T[gnr]['volm'])
            
            # acoustic information
            gnr_id = ptn_obj_dict[ptn]['gnr_ind'][gnr]
            gnr_acst = ptn_obj_dict[ptn]['acst_mat'][gnr_id]
            gnr_acsts[:,rel_ptns.index(ptn)]= gnr_acst

            # parent (prnt) information
            g_kld2 = ptn_obj_dict[ptn]['g_kld2']
            g_kld2_vd, g_kld2_vd_rv = vd_fer(g_kld2, g_kld2.vp.id)
            prnt_nds = list(g_kld2.vertex(g_kld2_vd[gnr]).in_neighbors())
            prnt_ids = [g_kld2.vp.id[i] for i in prnt_nds]
            gnr_prnts.append(prnt_ids)

            # children (chirn) information
            chirn_nds = list(g_kld2.vertex(g_kld2_vd[gnr]).out_neighbors())
            chirn_ids = [g_kld2.vp.id[i] for i in chirn_nds]
            gnr_chirn.append(chirn_ids)

            # already present variables
            df_res = ptn_obj_dict[ptn]['df_res']
            for vrbl in ord_vars:
                ord_var_dict[vrbl].append(df_res.T[gnr][vrbl])

        # sd might not be super meaningful because number of genres differs a lot between partitions
        # meaningful tho for metrics that don't depend on the sizes, which are still quite many
        ord_var_prcsd = {}
        for i in ord_vars:
            ord_var_mean = i+'_mean'
            ord_var_sd = i + '_sd'
            ord_var_max = i + '_max'
            ord_var_min = i + '_min'
            
            tual_vlus = weighted_avg_and_std(np, ord_var_dict[i], ptn_waets)

            ord_var_prcsd[ord_var_mean] = tual_vlus[0]
            ord_var_prcsd[ord_var_sd] = tual_vlus[1]
            ord_var_prcsd[ord_var_max] = max(ord_var_dict[i])
            ord_var_prcsd[ord_var_min] = min(ord_var_dict[i])
            

        prent = len(rel_ptns)

        acst_cor = np.corrcoef(gnr_acsts.T)
        if prent > 1:
            cor_vlus = acst_cor[np.where(np.triu(acst_cor,k=1)> 0)]
        else:
            cor_vlus = acst_cor

        mean_cor = np.mean(cor_vlus)
        cor_sd = np.std(cor_vlus)

        prnt_sims = np.mean(mult_dice(gnr_prnts))
        chirn_sims = np.mean(mult_dice(gnr_chirn))

        ord_var_prcsd['acst_cor_mean'] = mean_cor
        ord_var_prcsd['acst_cor_sd'] = cor_sd
        ord_var_prcsd['prnt_sims'] = prnt_sims
        ord_var_prcsd['chirn_sims'] = chirn_sims
        ord_var_prcsd['nbr_ptns'] = prent

        gnr_proc_dict[gnr] = ord_var_prcsd

    df_ttl = pd.DataFrame(gnr_proc_dict).T

    df_ttl['d1'] = d1
    df_ttl['d2'] = d2
    df_ttl['tp_id'] = tp_id

    return(df_ttl)



# * actual program

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('harsh_coef', help= 'how strongly to increase/decrease base thresholds')
    parser.add_argument('tp_coef', help= 'how strongly to increase/decrease time frame')
    parser.add_argument('tp_start', help= 'where to start tp')
    
    # parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()
    # x = args.file.readlines()
    

    # harsh_coef = 1.25
    # tp_coef = 0.75
    # res_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust/debug'


    harsh_coef = float(args.harsh_coef)
    tp_coef = float(args.tp_coef)
    tp_start = int(args.tp_start)

    
    # base values
    res_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust/harsh_' +str(harsh_coef) + "_tp_" + str(tp_coef) + '/'
    nr_wks = 16 * tp_coef
    min_cnt = 8 * harsh_coef * tp_coef
    min_weight = 16 * harsh_coef
    min_rel_weight = 0.16 * harsh_coef
    # min_tag_aprnc = 16 * harsh_coef
    # min_inst_cnt = 20 * harsh_coef
    # min_unq_artsts = 8 * harsh_coef
    
    min_tag_aprnc = 12 * harsh_coef * tp_coef
    min_inst_cnt = 16 * harsh_coef * tp_coef
    min_unq_artsts = 6 * harsh_coef * tp_coef

    # usr_dedcgs = 6 * harsh_coef
    # tag_plcnt = 16 * harsh_coef
    # unq_usrs = 16 * harsh_coef
    usr_dedcgs = 4 * harsh_coef * tp_coef
    tag_plcnt = 10 * harsh_coef * tp_coef
    unq_usrs = 10 * harsh_coef * tp_coef
    
    # base values have to be scaled also by tp_coef to get proportional levels
    # e.g. min_cnt 8: harsh_coef = tp_coef =1: has 16 weeks
    # to get the same difficulty with tp_coef = 1.25, min_cnt8 has to be scaled with it
    # i think i won't scale the time-invariant (min weight, min_rel_weight tho)
    # artists/usrs counts are not really invariant tho: have more time to generate them -> punish
    
    # and if not, i use what i have
    

    print('res_dir: ', res_dir,
          'nr_wks: ',         nr_wks, '\n',        
          'min_cnt: ',        min_cnt,        '\n',        
          'min_weight: ',     min_weight,     '\n',        
          'min_rel_weight: ', min_rel_weight, '\n',        
          'min_tag_aprnc: ',  min_tag_aprnc,  '\n',        
          'min_inst_cnt: ',   min_inst_cnt,   '\n',        
          'min_unq_artst: ',  min_unq_artsts,  '\n',        
          'usr_dedcgs: ',     usr_dedcgs,     '\n',        
          'tag_plcnt: ',      tag_plcnt,      '\n',        
          'unq_usrs: ',       unq_usrs)

    # res_dir = arg_dict['res_dir']
    # nr_wks = arg_dict['nr_wks']
    # min_cnt = arg_dict['min_cnt']
    # min_weight = arg_dict['min_weight']
    # min_rel_weight = arg_dict['min_rel_weight']
    # min_tag_aprnc = arg_dict['min_tag_aprnc']
    # min_inst_cnt = arg_dict['min_inst_cnt']
    # min_unq_artsts = arg_dict['min_unq_artsts']
    # usr_dedcgs = arg_dict['usr_dedcgs']
    # tag_plcnt = arg_dict['tag_plcnt']
    # unq_usrs = arg_dict['unq_usrs']

    max_propx1 = 0.5
    max_propx2 = 0.7
    ptn = '_all'


    time_periods = gnr_t_prds(7*nr_wks)
    print(time_periods[0], time_periods[-1], len(time_periods))

    vrbls=['dncblt','gender','timb_brt','tonal','voice','mood_acoustic',
           'mood_aggressive','mood_electronic','mood_happy','mood_party','mood_relaxed','mood_sad'] 
    
    # tprd = time_periods[0]
    
    print(len(time_periods))
    if tp_start == len(time_periods):
        print('done here')
        raise SystemExit


    for tprd in time_periods[tp_start:]:
        print(tprd)
        
        client = Client(host='localhost', password='anudora', database='frrl')
        d1 = tprd[0].strftime('%Y-%m-%d')
        d2 = tprd[1].strftime('%Y-%m-%d')
        d1_dt = datetime.strptime(d1, '%Y-%m-%d')
        d2_dt = datetime.strptime(d2, '%Y-%m-%d')
        base_dt = datetime(1970, 1, 1)
        d1_int = (d1_dt - base_dt).days
        d2_int = (d2_dt - base_dt).days
        tp_id = time_periods.index(tprd)
        tp_clm = d1 + ' -- ' + d2

        ptns = ['_all']
        ptn_obj_dict = {}
        client = Client(host='localhost', password='anudora', database='frrl')

        for ptn in ptns:
            ptn_obj_dict[ptn] = ptn_proc(ptn)
        
        df_ttl = ptn_obj_dict[ptn]['df_res']

        df_ttl['d1'] = d1
        df_ttl['d2'] = d2
        df_ttl['tp_id'] = tp_id

        # df_ttl = ptn_eval(ptns, ptn_obj_dict)
        df_ttl.to_csv(res_dir + tp_clm + '.csv')
        

        # pnt_obj_dict[ptn] = {}

        # print(df_res.shape)
        # print('print to csv')
        # df_res.to_csv(res_dir + tp_clm + '.csv')

        # raise Exception('done')



# break


    

# * xprtl

# ** ptn partitioning sketch
#     # gnrs = list(np.unique(dfc['tag']))
#     # ptn_gnrs.append(gnrs)

# # gnr comparison
# NBR_PTNS =3

# ptn_ovlp = np.zeros((NBR_PTNS,NBR_PTNS))
# for i1 in range(NBR_PTNS):
#     i1_ptn = ptn_gnrs[i1]
#     for i2 in range(NBR_PTNS):
#         i2_ptn = ptn_gnrs[i2]

#         ovlp = len(list(set(i1_ptn) & set(i2_ptn)))
#         ptn_ovlp[i1,i2] = ovlp

# all_gnrs = set(list(itertools.chain.from_iterable(ptn_gnrs)))



# np.corrcoef(tri_vlus1, tri_vlus2)



# ** tag_song network

# trk_gnr_el = [i for i in zip(dfc['lfm_id'], dfc['tag'], dfc['rel_weight'], dfc['cnt'], dfc['rel_weight']*dfc['cnt'])]

# g_trks = Graph()
# g_trks_waet = g_trks.new_edge_property('float')
# g_trks_cnt = g_trks.new_edge_property('float')
# g_trks_sz = g_trks.new_edge_property('float')

# g_trks_id = g_trks.add_edge_list(trk_gnr_el, hashed = True, string_vals = True, eprops = [g_trks_waet, g_trks_cnt, g_trks_sz])
# g_trks_vd, g_trks_id_rv = vd_fer(g_trks, g_trks_id)

# gnr = 'alternative'
# # comparing genres is easier than for loop

# gnr_comps = all_cmps_crubgs(gnrs, g_trks_vd, 'product')

# gnr_sims = vertex_similarity(GraphView(g_trks, reversed = True), 'dice', vertex_pairs = gnr_comps, eweight = g_trks_waet)

# gnr_sim_ar = np.array(np.split(gnr_sims, len(gnrs)))

# gnr_sim_ar2 = asym_sim2(g_trks, g_trks_vd, gnrs, gnr_sim_ar, g_trks_waet)

# # len(gnr_sim_ar2[np.where(gnr_sim_ar2 > 0.3)]) # 1845
# nph(gnr_sim_ar2[np.where(gnr_sim_ar2 > 0.3)][np.where(gnr_sim_ar2[np.where(gnr_sim_ar2 > 0.3)] < 1)])

# nph(gnr_sim_ar2[np.where(gnr_sim_ar2 > 0.01)])

# # look what alternative is actually similar to
# # am i looking at to or from? 

# # gnr_sim_ar2[:,gnr_ind['alternative']][
    
# # stuff that is similar to alternative
# simsTo = np.where(gnr_sim_ar2[:,gnr_ind['alternative']] > 0.05)
# [print(gnrs[i]) for i in simsTo[0]]

# # stuff alternative is similar to
# # it's actually weird: alternative is so big, it's kinda surprising that there is that much that alternative is similar to
# # maybe it's really the other way around?  yeaaaah pretty sure 
# simsFrom = np.where(gnr_sim_ar2[gnr_ind['alternative']] > 0.2)
# [print(gnrs[i]) for i in simsFrom[0]]


# # welp at least i don't have the problem of undiscriminatorily high similarity values FUCK ME
# # add weights to asym_sim2 and nothing substantially over 1; lovely



# ** different clusterings
# *** AHC of acst_usr mat

# # cosine_similarity is fast at least, 8m/sec
# # would take 6sec for 10k users

# # not clear how long acoustic mat construction would take
# # CH has quantile functions, not sure how fast they are


# usr_qnt_tbl = """CREATE TEMPORARY TABLE usr_qntls (
#     usr String,
#     lfm_id String,
#     cnt UInt16, 
#     """ + ",\n".join([i + ' Float32' for i in vrbls])+ ")"


# d1 = '2010-08-28'
# d2 = '2010-11-20'
# vrbl_strs  = ", ".join(vrbls)


# usr_qnt_insert = """
# INSERT INTO usr_qntls
# SELECT * FROM (
#     SELECT usr, mbid as lfm_id, cnt FROM (
#         SELECT usr, song as abbrv, count(usr,song) as cnt FROM logs
#         WHERE time_d BETWEEN '"""  + d1 + """' and '""" + d2 + """'
#         GROUP BY (usr,song)
#     ) JOIN (SELECT mbid, abbrv FROM song_info) USING abbrv
# ) JOIN (SELECT lfm_id, """ + vrbl_strs +""" FROM acstb) USING lfm_id"""


# client.execute('drop table usr_qntls')
# client.execute(usr_qnt_tbl)
# client.execute(usr_qnt_insert)

# # client.execute('select count(*) from usr_qntls')

# # nph(some_vlus)
# # npl(ch_hist[0][0])

# # nps(ch_hist[0][0], range(11), 1)

# # x = pd.DataFrame(ch_hist[0][0], columns = ['asdf'])


# # maybe manual hist is easier?
# # count(songs) * cnt _cnt group by user divided by ttl  count for user
# # problem is then to account for users who have no values and therefore don't get added to grouping

# NBR_CHNKS = 9

# qnt_brdrs = [(i,i + 1/NBR_CHNKS) for i in np.arange(0,1,1/NBR_CHNKS)]

# # using 10 chunks probably results in song duplicates

# unq_usrs = client.execute('SELECT DISTINCT usr from usr_qntls')
# unq_usrs = [i[0] for i in unq_usrs]

# qnt_dict = {}
# for u in unq_usrs:
#     qnt_dict[u] = []

# for vrbl in vrbls:

#     for brd in qnt_brdrs:
#         prent_usrs = []

#         qnt_qry = """SELECT usr, sum(cnt) from usr_qntls 
#         where """ + vrbl + """ BETWEEN """ + str(brd[0]) + " and " + str(brd[1]) + " GROUP BY usr"

#         qnt_vlus = client.execute(qnt_qry)
#         for qv in qnt_vlus:
#             qnt_dict[qv[0]].append(qv[1])
#             prent_usrs.append(qv[0])

#         # handle missing users
#         mis_usrs = set(unq_usrs) - set(prent_usrs)
#         for mu in mis_usrs:
#             qnt_dict[mu].append(0)



# usr_acst_ar = pd.DataFrame(qnt_dict).T
# sums = usr_acst_ar.sum(axis = 1)/len(vrbls)

# sum_ar = np.array([sums] * len(vrbls)*NBR_CHNKS).T

# usr_acst_probs = usr_acst_ar/sum_ar



# from sklearn.metrics.pairwise import euclidean_distances

# x = cosine_similarity(usr_acst_probs)
# x = euclidean_distances(usr_acst_probs)
# x[np.where(x == 0)] = 0.001

# dist_mat = -np.log(x)
# nph(dist_mat)


# from sklearn.cluster import AgglomerativeClustering
# from collections import Counter

# cluster = AgglomerativeClustering(n_clusters = 8, affinity='precomputed', linkage ='complete')
# clstrs = cluster.fit_predict(dist_mat)
# clstrs_eucld = cluster.fit_predict(x)
# Counter(clstrs)
# Counter(clstrs_eucld)

# # fuck everything too similar
# # how can it be that users most users who have less than 4% of songs in common have basically the same sound profile
# # because all music sounds the same? 

# # maybe use some minimum amount of songs that persons needs? not much impact

# # cluster overlap matrix? think i did that
# # yup, only one cluster
# # maybe there really is just one?
# # but how to explain the graphtool clustering then?
# # it is reliable over multiple clustering, so not a random partition of one group
# # maybe gt clustering is just the only way to go?
# # see if gt clustering can be tweaked to be faster



# *** cluster users based on the tags

# usr_tag_tbl = """CREATE TEMPORARY TABLE usr_tag_tbl (
# usr String,
# tag String,
# vol Float32)"""

# usr_tag_qry = """
# INSERT INTO usr_tag_tbl SELECT usr, tag, sum(vol) FROM (
#     SELECT usr, tag, cnt*rel_weight as vol FROM (
#         SELECT usr, song as abbrv, count(usr, song) as cnt FROM logs
#         WHERE time_d BETWEEN '""" + d1 + """' AND '""" + d2 + """'
#         GROUP BY (usr, song)
#     ) JOIN (
#         SELECT mbid, abbrv, tag, rel_weight FROM (
#             SELECT mbid, tag, rel_weight FROM tag_sums
#             JOIN (
#                 SELECT tag FROM tag_sums WHERE rel_weight > 0.075 
#                 GROUP BY tag
#                 HAVING count(tag) > 40
#             ) USING tag
#         )
#         JOIN (SELECT mbid, abbrv FROM song_info) USING mbid
#         WHERE rel_weight > """ + str(min_rel_weight) + """
#     ) USING abbrv
# ) GROUP BY (usr, tag)"""

# # tag_sums

# print(usr_tag_qry)
# client.execute('drop table usr_tag_tbl')
# client.execute(usr_tag_tbl)
# client.execute(usr_tag_qry)

# usr_tag_lnks = client.execute('SELECT * FROM usr_tag_tbl')
# unq_usrs = client.execute('select distinct usr from usr_tag_tbl')
# unq_usrs = [i[0] for i in unq_usrs]

# g_usr_tag = Graph()
# usr_tag_vol = g_usr_tag.new_edge_property('float')

# g_usr_tag_id = g_usr_tag.add_edge_list(usr_tag_lnks, hashed=True, string_vals=True, eprops = [usr_tag_vol])
# g_usr_tag_vd, g_usr_tag_vd_rv = vd_fer(g_usr_tag, g_usr_tag_id)

# unq_usrs_ids = [g_usr_tag_vd[i] for i in unq_usrs]
# usr_comps = list(itertools.combinations(unq_usrs_ids, 2))

# smpl_sims = vertex_similarity(g_usr_tag, 'jaccard', vertex_pairs = usr_comps, eweight = usr_tag_vol)


# N_SAMPLE = len(unq_usrs)
# tri = np.zeros((N_SAMPLE, N_SAMPLE))
# tri[np.triu_indices(N_SAMPLE, 1)] = smpl_sims
# tri.T[np.triu_indices(N_SAMPLE, 1)] = smpl_sims

# nph(tri)

# dist_mat = -np.log(tri)
# actual_max = np.max(dist_mat[np.where(dist_mat < math.inf)])
# dist_mat[np.where(dist_mat > actual_max)] = actual_max + 2

# nph(dist_mat)

# from sklearn.cluster import AgglomerativeClustering
# cluster = AgglomerativeClustering(n_clusters = 5, affinity='precomputed', linkage ='complete')
# clstrs = cluster.fit_predict(dist_mat)
# Counter(clstrs)

# # somewhat better (get two clusters with 5, but stillmany super small ones)


# ** more GT: tweak settings

# GT operates on massively trimmed graph in that most people are assumed to be unconnected
# is that justifiable? 
# is that transferable to other forms? 


# * scrap
# ** time durations

# # ch_qry = 'SELECT time_d, count(time_d) FROM logs GROUP BY time_d'
# # time_cnts = client.execute(ch_qry)
# # time_pnts = [i[0] for i in time_cnts]
# # cnts = [i[1] for i in time_cnts]

# # ax = plt.axes()
# # ax.plot(time_pnts, cnts)
# # plt.show()

# # qry = 'select time_d, uniq(usr) from logs group by time_d'
# # time_cnts = client.execute(qry)
# # time_pnts = [i[0] for i in time_cnts]
# # unq_usrs = [i[1] for i in time_cnts]

# # ax = plt.axes()
# # ax.plot(time_pnts, unq_usrs)
# # plt.show()


# ** speed up implementations
# *** dict_gnrs: not primarily important

# # is dict_gnrs (produces acst_gnr_dict) parallelizable?
# # in principle yeah: can split pandas df, process rows separately,
# # merging into one in the end is a bit work but not too much tbh

# # takes 10 sec, 30sec for 6 months
# # idk that's like 10% of the time (5 min), but kinda neglible against
# # el_acst_mp: even with multiprocessing still 46 second -> 90 sec saved
# # kld time: even when paralellized, still takes for fucking ever: 250 sec,
# # tbh firefox took up a lot but still



# *** fucking done
# # gnrt_acst_el is single core, can be parallelized tho, might be worth it
# # kld mat also takes quite some time
# # wonder if custom cython function would be faster
# # seems to be already heavily using C funcs, so don't really think there's much to improve

# # KLD is fucking fast with broadcasting
# # feature extraction also parallelized


# ** trying to use CH functionality for usr histogram, but not working
# how to convert CDF to histogram? 
# relative change?
# i want buckets/cutoffs

# i now know there is the 
# SELECT usr, quantilesTDigestWeighted(0, 0.2, 0.4, 0.6, 0.8, 1)(dncblt,cnt) FROM usr_qntls GROUP BY usr

# some_vlus = client.execute("select dncblt from usr_qntls where usr = 'f18621'")
# ch_hist = client.execute("SELECT quantilesTDigestWeighted(0, 0.2, 0.4, 0.6, 0.8, 1)(dncblt,1) FROM usr_qntls where usr = 'f18621'")

# ch_hist = client.execute("SELECT quantilesExactWeighted(0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1)(dncblt,1) FROM usr_qntls where usr = 'f18621'")


# * test alternative cells


# acst_mat2 = acst_clr_mp(gnrs, acst_gnr_dict, gnr_ind)
# ar_cb2 = acst_cpr2_mp_mng(gnrs, gnr_ind, acst_mat2)

# npr = 3
# kld3_el = kld_n_prnts(ar_cb, npr, gnrs, gnr_ind)
# kld3_el = kld_n_prnts2(ar_cb2.T, npr, gnrs, gnr_ind)
# g_kld2, vd_kld2, vd_kld2_rv = kld_proc(kld3_el)

# graph_pltr(g_kld2, g_kld2.vp.id, 'cell_combns.pdf', 0.4)

# # ar_cb2 needs to be T-ed
# # kld_n_prnts needs to be adapted to handle lack of parents



# t1 = time.time()
# test_res = acst_cpr2(gnrs[3], gnrs[0:30])
# t2 = time.time()

# x = acst_cpr2(gnr_cprs[30])[1]

# 264 comps per sec
# so like 1k/sec with multiprocessing
# could do a whole lot of g1s for each g2 zeroes
# proper sum: 12k/sec, seems acceptable with multiprocessing
# can increase it by another factor of ~2 by keepign g2_zeros constant
# so i basically keep parent constant? so i basically calculate rows in ar_cb

## can't broadcast that shit


# * network stats
# # pagerank
# pgrnk = pagerank(g_kld2)
# pgrnks = [pgrnk[vd_kld2[i]] for i in gnrs]

# # betweenness
# tweens = betweenness(g_kld2)[0]
# tween_vlus = [tweens[vd_kld2[i]] for i in gnrs]
# # might be a kind of brokering? but only non-zero for like half: pretty sure none with that will die out
# # tons of zeroes


# # closeness: as measure of centrality? 
# # many nans tho
# clsns = closeness(g_kld2)
# clsns_vlus = [clsns[vd_kld2[i]] for i in gnrs]

# would also have to figure out how to use weights

# # i don't like spaces
# # could actually have been that the decisive calculations were done with 1cell

# eig_vec = eigenvector(g_kld2, g_kld2.ep.kld_sim)[1]
# eig_vecs = [eig_vec[vd_kld2[i]] for i in gnrs]
# # seems to produce nonsense

# katz_vp = katz(g_kld2)
# katz_vlus2 = [katz_vp[vd_kld2[i]] for i in gnrs]

# # nonsense




# ar_cb2 = kld_schema_mp(vrbls, acst_mat, nbr_cls)

# ## weights don't seem to add up amazingly


# npr = 3
# kld2_el = kld_n_prnts(kld_schema, npr, gnrs, gnr_ind)

# print('construct kld graph')
# g_kld2, vd_kld2, vd_kld2_rv = kld_proc(kld2_el)
# graph_pltr(g_kld2, g_kld2.vp.id, '1_cell_space.pdf', 0.3)




# [print(gnrs[i], dncblt_mat[i]) for i in list(np.where(dncblt_diagnst > 0.8)[0])]
# seems to be mostly diagnostic for undanceable ones




## how to use diagnostic weights? 
## i have two comparisons
## can make quadratic diag weight mat and use the weights of both genres in both directions
## if i compare objects a and b, it seems like it should matter how important how important the feature is for both objects




   
# ** general comparison

## entropy is somewhow normalized, don't think it really makes a difference tho

# v1 = acst_mat[gnr_ind['ambient']]
# v2 = acst_mat[gnr_ind['jazz']]

# vlus = []
# for vrbl in vrbls: 
#     rel_cols = np.where(np.array(vrbl_vec) == vrbl)[0]
#     v1_mod = v1[rel_cols]
#     v2_mod = v2[rel_cols]
    
#     vlu = entropy(v1_mod,v2_mod)
#     vlus.append(vlu)
    

# entropy(v1, v2)


# ar_cb[gnr_ind['jazz'], gnr_ind['ambient']]

# vrbl = 'dncblt'


# acst_mat_sbst = acst_mat[:,rel_cols]

# kldx = entropy(acst_mat_sbst[:,:].T[:,:,None], acst_mat_sbst.T[:,None,:])

# all_acst = dfc[['lfm_id'] + vrbls].groupby('lfm_id').max()

# cor_mat = np.corrcoef(all_acst.T)
# plt.matshow(cor_mat)
# plt.show()

# highest is 0.78 between mood acoustic and mood_sad
# 




    
# * clean up ptn_proc
# print('construct acoustic edge list')
# el_ttl = gnrt_acst_el_mp(gnrs, acst_gnr_dict, nbr_cls)
# print('construct acoustic graph')
# gac, w, w_std, w_std2, gac_id, vd, vdrv = gac_crubgs(el_ttl)
# print('construct acoustic mat')
# acst_mat = acst_arfy(el_ttl, vrbls, 3, gnrs, nbr_cls)

# ** clean up actual program partition stuff

# CREATE PARTITIONS
# min_usr_cnt = 25
# song has to be listened to by at least that many users
# min_usr_plcnt = 50
# user has to play at least that many (unique) songs (which in turn have at least min_usr_cnt

# d1 = str(time_periods[0][0])
# d2 = str(time_periods[-1][1])
# ptn_vars = " ".join([str(i) for i in [d1, d2, min_cnt, min_usr_cnt, min_usr_plcnt]])
# ptn_str = 'python3.6 ptn_lda.py ' + ptn_vars
# os.system(ptn_str)

# ptn_str = 'python3.6 ptn_lda2.py ' + d1 + ' ' + d2
# os.system(ptn_str)

# ptns = list(range(5))

# * parent check
# np.corrcoef(df_res_pr4['smpl_ftrs_prnt3_dvrg'], df_res_pr2['smpl_ftrs_prnt3_dvrg'])
# np.corrcoef(df_res_pr4['smpl_ftrs_cohrt_mean_non_inf_wtd'], df_res_pr2['smpl_ftrs_cohrt_mean_non_inf_wtd'])
# np.corrcoef(df_res_pr4['smpl_ftrs_cohrt_len'], df_res_pr2['smpl_ftrs_cohrt_len'])
