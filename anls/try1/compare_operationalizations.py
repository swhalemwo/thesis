
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



os.chdir("/home/johannes/Dropbox/phd/papers/genres/anls/try1/add_data/")

print(os.getcwd())
from gnrl_funcs import get_dfs
from gnrl_funcs import dict_gnrgs
from acst_hier import gnr_t_prds
from datetime import date, datetime
from acst_hier import gnrt_sup_dicts, split
from acst_hier import krnl_acst, krnl_acst_mp
from acst_hier import kld_mat_crubgs, kld_mp2
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

    tprd = [date(2011,1,1), date(2011,12,31)]


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


dfc = get_dfc()

gnrs = list(np.unique(dfc['tag']))
artsts = list(np.unique(dfc['artist']))
trks = list(np.unique(dfc['lfm_id']))

nbr_cls = 7
acst_gnr_dict = dict_gnrgs(dfc, gnrs, pd)
sz_dict, gnr_ind, waet_dict, vol_dict = gnrt_sup_dicts(acst_gnr_dict, gnrs)

acst_mat = krnl_acst_mp(gnrs, acst_gnr_dict, nbr_cls, vrbls)

ar_cb = kld_mat_crubgs(gnrs, acst_mat)


