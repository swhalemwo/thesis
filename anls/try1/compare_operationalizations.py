
import os
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
from graph_tool.all import *
from graph_tool import *

from dotmap import DotMap as ddict
# https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
# https://github.com/drgrib/dotmap
# should use dot dictionaries for config stuff, much more convenient to type


os.chdir("/home/johannes/Dropbox/phd/papers/genres/anls/try1/add_data/")

print(os.getcwd())
from gnrl_funcs import get_dfs
from gnrl_funcs import dict_gnrgs
from acst_hier import gnr_t_prds
from datetime import date, datetime
from acst_hier import gnrt_sup_dicts, split, vd_fer
from acst_hier import krnl_acst, krnl_acst_mp
from acst_hier import kld_mat_crubgs, kld_mp2
from acst_hier import kld_n_prnts
from acst_hier import kld_proc
from acst_hier import asym_sim2
from aux_funcs import nph, render_graph_with_graphviz
from acst_hier import all_cmps_crubgs
from acst_hier import gini

from scipy.stats import entropy
from scipy.special import rel_entr
from sklearn.neighbors import KernelDensity



vrbls=['dncblt','gender','timb_brt','tonal','voice','mood_acoustic',
           'mood_aggressive','mood_electronic','mood_happy','mood_party','mood_relaxed','mood_sad'] 



def get_dfc():
    """get the pandas data frame, for now settings kinda arbitrary/unsystematic"""

    client = Client(host='localhost')

    # print(time_periods[0], time_periods[-1], len(time_periods))

    vrbls=['dncblt','gender','timb_brt','tonal','voice','mood_acoustic',
           'mood_aggressive','mood_electronic','mood_happy','mood_party','mood_relaxed','mood_sad'] 

    tprd = [date(2011,1,1), date(2011,3,31)]


    d1 = tprd[0].strftime('%Y-%m-%d')
    d2 = tprd[1].strftime('%Y-%m-%d')
    d1_dt = datetime.strptime(d1, '%Y-%m-%d')
    d2_dt = datetime.strptime(d2, '%Y-%m-%d')
    base_dt = datetime(1970, 1, 1)
    d1_int = (d1_dt - base_dt).days
    d2_int = (d2_dt - base_dt).days
    # tp_id = time_periods.index(tprd)
    tp_clm = d1 + ' -- ' + d2


    min_cnt = 20
    min_weight = 20
    min_rel_weight = 0.2
    min_tag_aprnc = 0.2
    min_inst_cnt = 20

    min_unq_artsts = 10

    usr_dedcgs = 6
    tag_plcnt = 10

    unq_usrs = 10                   

    max_propx1 = 0.5
    max_propx2 = 0.7
    ptn = '_all'



    dfc = get_dfs(vrbls, min_cnt, min_weight, min_rel_weight, min_tag_aprnc,
                  min_unq_artsts, max_propx1, max_propx2, d1, d2, ptn,
                  usr_dedcgs, tag_plcnt, unq_usrs,
                  client, pd)


    return dfc


def get_cooccurence_mat(dfc, gnrs):
    """create asymmetric co-occurence matrix"""
    trk_gnr_el = [i for i in zip(dfc['lfm_id'], dfc['tag'], dfc['rel_weight'], dfc['cnt'], dfc['rel_weight']*dfc['cnt'])]

    g_trks = Graph()
    g_trks_waet = g_trks.new_edge_property('float')
    g_trks_cnt = g_trks.new_edge_property('float')
    g_trks_sz = g_trks.new_edge_property('float')

    g_trks_id = g_trks.add_edge_list(trk_gnr_el, hashed = True, eprops = [g_trks_waet, g_trks_cnt, g_trks_sz])
    g_trks_vd, g_trks_id_rv = vd_fer(g_trks, g_trks_id)

    gnr = 'alternative'
    # comparing genres is easier than for loop

    gnr_comps = all_cmps_crubgs(gnrs, g_trks_vd, 'product')

    gnr_sims = vertex_similarity(GraphView(g_trks, reversed = True), 'dice', vertex_pairs = gnr_comps, eweight = g_trks_waet)

    gnr_sim_ar = np.array(np.split(gnr_sims, len(gnrs)))

    gnr_sim_ar2 = asym_sim2(g_trks, g_trks_vd, gnrs, gnr_sim_ar, g_trks_waet)

    return gnr_sim_ar2


def describe_graph(g):
    """generate all kinds of descriptives about graph"""

    results = {}
    # global clustering also returns also standard variation
    global_clust_coef = graph_tool.clustering.global_clustering(g)
    results['global_clust_coef'] = global_clust_coef[0]
    results['global_clust_coef_sd'] = global_clust_coef[1]
    
    results['edge_reciprocity'] = graph_tool.topology.edge_reciprocity(g)

    g.set_directed(False)
    dist = graph_tool.topology.shortest_distance(g)
    avg_path_length = sum([sum(i) for i in dist])/(g.num_vertices()**2-g.num_vertices())
    results['avg_path_length'] = avg_path_length
    g.set_directed(True)

    # concentration measures of degree distribution
    deg_map = g.degree_property_map("out")
    deg_ar = deg_map.get_array()

    results['gini_out_deg'] = gini(deg_ar).tolist()

    top10_percentile = np.percentile(deg_ar, 90)
    results['top10_perc_conc'] = deg_ar[np.where(deg_ar >= top10_percentile)].sum().tolist() / deg_ar.sum().tolist()

    
    # could add weights: but KDL and overlap have different distributions

    return results
    


def get_kld_dist(dfc):
    """process the dfc into asymmetric distance matrix"""

    gnrs = list(np.unique(dfc['tag']))
    artsts = list(np.unique(dfc['artist']))
    trks = list(np.unique(dfc['lfm_id']))

    nbr_cls = 7
    acst_gnr_dict = dict_gnrgs(dfc, gnrs, pd)
    sz_dict, gnr_ind, waet_dict, vol_dict = gnrt_sup_dicts(acst_gnr_dict, gnrs)

    acst_mat = krnl_acst_mp(gnrs, acst_gnr_dict, nbr_cls, vrbls)

    ar_cb = kld_mat_crubgs(gnrs, acst_mat)

    # create KLD-based graph
    npr = 3
    kld2_el = kld_n_prnts(ar_cb, npr, gnrs, gnr_ind)
    g_kld2, vd_kld2, vd_kld2_rv = kld_proc(kld2_el)
    kld = {'g':g_kld2, 'vd':vd_kld2, 'vdrv': vd_kld2_rv}

    # create overlap based graph
    gnr_sim_ar2 = get_cooccurence_mat(dfc, gnrs)
    ovlp_el = kld_n_prnts(1-gnr_sim_ar2, npr, gnrs, gnr_ind)
    g_ovlp, vd_ovlp, vd_ovlp_rv = kld_proc(ovlp_el)
    ovlp = {'g':g_ovlp, 'vd': vd_ovlp, 'vdrv': vd_ovlp_rv}

    return kld, ovlp

dfc = get_dfc()

kld, ovlp = get_kld_dist(dfc)

# see how much edges of graphs overlap: need to adjust now, but maybe useful in comparison to established hierarchies
kld_el_basic = [(i[0], i[1]) for i in kld2_el]
ovlp_el_basic = [(i[0], i[1]) for i in ovlp_el]
len(set(ovlp_el_basic).intersection(set(kld_el_basic)))/len(ovlp_el_basic)
# just 200/10% of links the same huh

graph_pltr2(kld['g'], "/home/johannes/Dropbox/phd/papers/genres/figures/kld_test.pdf", g_kld2.ep.kld_sim)


graphviz_draw(g_kld2,
              output = "/home/johannes/Dropbox/phd/papers/genres/figures/kld_test.pdf")

FIG_DIR = "/home/johannes/Dropbox/phd/papers/genres/figures/"
render_graph_with_graphviz(kld['g'], FIG_DIR + 'kld.pdf')
render_graph_with_graphviz(ovlp['g'], FIG_DIR + 'kld.pdf')




    




