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

# * hierarchical relations based on split dimensions
# functions defined in item_tpclt



def nph(ar_x):
    """custom hist function because plt.hist is fucking unreliably"""
    a1,a2 = np.histogram(ar_x, bins='auto')
    # width = 0.7 * (a2[1] - a2[0])
    width = (a2[1] - a2[0])
    center = (a2[:-1] + a2[1:]) / 2
    plt.bar(center, a1, align='center', width=width)
    plt.show()


g = g_kld2
ids = g_kld2_id
filename = 'acst_spc5.pdf'
eweit = kld_sim_int

def graph_pltr(g, ids, filename, eweit):
    """function for graph plotting, maybe put all the plotting parameters into function too?"""

    gt_lbls_plot = g.new_vertex_property('string')

    for v in g.vertices():
        x = ids[v]

        gt_lbls_plot[v] = x.replace(" ", "\n")

    size = g.degree_property_map('out')

    # size_scl=graph_tool.draw.prop_to_size(size, mi=4, ma=8, log=False, power=0.5)
    size_scl=graph_tool.draw.prop_to_size(size, mi=7, ma=25, log=False, power=0.5)

    size_scl2=graph_tool.draw.prop_to_size(size, mi=0.025, ma=0.15, log=False, power=1)

    if type(eweit) == type (1):
        e_scl = eweit
        pass
    else:
        e_scl=graph_tool.draw.prop_to_size(eweit, mi=1, ma=6, log=False, power=0.5)

    gvd = graphviz_draw(g, size = (50,50),
                        # layout = 'sfdp',
                        # overlap = 'scalexy',
                        overlap = 'false',
                        vprops = {'xlabel':gt_lbls_plot, 'fontsize':size_scl, 'height':0.03,
                                  'shape':'point', 'fixedsize': True,
                                  'width':size_scl2, 'height':size_scl2, 'fillcolor':'black'},
                        eprops = {'arrowhead':'vee', 'color':'grey', 'weight':eweit,
                                  'penwidth':e_scl},
                        # returngv==True,
                        output = filename)
    gt_lbls_plot = 0


def vd_fer(g, idx):
    """creates vertex index dict from graph and id property map"""
    # vertex_dict
    vd = {}
    vdrv = {}
    for i in g.vertices():
        vd[idx[i]] = int(i)
        vdrv[int(i)] = idx[i]
        
    return(vd, vdrv)

# gnrs = ['metal', 'ambient', 'rock', 'hard rock', 'electronic', 'death metal']*10
# gnrs = list(acst_gnr_dict.keys())


def gnrt_acst_el(acst_gnr_dict, gnrz):
    """generates edge list for acoustic space network"""
    
    # gnrz = acst_gnr_dict.keys()
    el_ttl = []
    sz_dict = {}
    
    for gnr in gnrz:
    #     print(gnr)
        
        # dfc = get_df_cbmd(gnr)
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
    return(el_ttl, sz_dict)

el_ttl, sz_dict = gnrt_acst_el(acst_gnr_dict, gnrs)

# fig = plt.figure()
# ax = plt.axes()
# # ax.plot(range(10),a_wtd)
# ax.plot(range(10),a_wtd_old)
# plt.show()

# graph acoustic n
gac = Graph()

# w = gac.new_edge_property('int16_t')
w = gac.new_edge_property('double')
w_std = gac.new_edge_property('double')
w_std2 = gac.new_edge_property('double')

idx = gac.add_edge_list(el_ttl, hashed=True, string_vals=True,  eprops = [w, w_std, w_std2])

vd,vdrv = vd_fer(gac, idx)

# * general comparison function

# order doesn't matter anymore due to standardization
# do i need a matrix?
# can read it back in (either lower triangle columnwise or upper rowise)
# not sure if needed tho
# can just the positions of those over threshold, get the corresponding original comparisions, get the corresponding genres, and add that to edge list

# nested loops to ensure direction
# smaller one is now first: relation to be tested is subsetness
# but there is no a priori reason why smaller genre should be subset
# subgenre can also grow larger than original (w40k > warhammer)

def cmp_crubgs(gnrs, vd):

    gnr_ids = [vd[i] for i in gnrs]
    
    lenx = len(gnrs)

    cprx = []
    c = 0
    for i in range(lenx):

        cprx2 = []

        for k in range(i,lenx):

            # print(k)

            if i==k:
                next
            else:
                c +=1

                v1, v2 = gnrs[i], gnrs[k]
                v1_sz, v2_sz = sz_dict[v1], sz_dict[v2]

                if v1_sz > v2_sz:
                    cprsn = [vd[v2], vd[v1]]
                else:
                    cprsn = [vd[v1], vd[v2]]

                cprx2.append(cprsn)
        if len(cprx2) > 0:
            cprx.append(cprx2)

    cmps = list(itertools.chain.from_iterable(cprx))
    return(cmps)

cmps = cmp_crubgs(gnrs, vd)

def all_cmps_crubgs(gnrs, vd, type): 
    gnr_ids = [vd[i] for i in gnrs]
    lenx = len(gnrs)

    if type == "permutations":
        cmps = list(itertools.permutations(gnr_ids, 2))

    if type == "product":
        cmps = list(itertools.product(gnr_ids, repeat=2))
        
    return(cmps)


cmps = all_cmps_crubgs(gnrs, vd, 'product')

# * actual comparision
# is not asymmetric now
# that's what happens when you standardize 
# could standardize so that max = 1,
# 3,3,2 -> 1,1,0.6
# 3,0.5,0.5 -> 1,1,0.1666
# scaling down with max, not total sum -> ask MARIEKE

# seems not too good: some pointless genres (00s, 10 stars become super large)

gt_sims = vertex_similarity(gac, 'dice', vertex_pairs = cmps, eweight=w_std2)

plt.hist(gt_sims, bins='auto')
plt.show()

asym_sim_ar = asym_sim(gnrs, vd)

nph(asym_sim_ar)




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


# Graph asymmetric hierarchy acoustic

el_acst = sbst_eler(asym_sim_ar, operator.gt, 0.99)

ghrac = Graph()
sim_vlu = ghrac.new_edge_property('double')
ghrac_id = ghrac.add_edge_list(el_acst, hashed=True, string_vals=True, eprops  = [sim_vlu])

# vertex dict hierachical
vd_hr, vd_hr_rv = vd_fer(ghrac, ghrac_id)

# does not reliable capture birirectionality
graph_pltr(ghrac, ghrac_id, 'acst_spc3.pdf')

graph_draw(ghrac, output='ghrac.pdf')

for v in ghrac.vertices():
    print(ghrac_id[v], v.out_degree(),v.in_degree() )

# g = ghrac
# ids = ghrac_id
# filename = 'acst_space1.pdf'

# no more artists, i guess that's something
# but not super happy with the lack of directionality


# * other comparisons
# if i treat the splitted stuff as features, why not cosine similarity?
# or correlation?
# if it's standardized it's symmetric now anyways
# FUUU?
# could do it all in matrices?
# Hannan not that bad?
# substantial differences is to split up into cells, not so much way to calculate then similarity
# footnote p. 50: are aware of that variance matters -> bring split up as improvement?
# also difference between measures -> correlation

# how fast is KLD for that many comps? 

# ** turn el into array/df


# x = np.split(np.array(el_ttl), 50)
# x2 = np.array(x)

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

acst_mat = acst_arfy(el_ttl, vrbls,3)

# 10k comparisons/sec, can be parallelized -> not too bad
# timeit('entropy(acst_mat[0], acst_mat[1])', globals=globals(), number=100000)

# ** KLD
# needs functionizing and parallel processing 
gnr_ind = {}
for i in gnrs:
    gnr_ind[i] = gnrs.index(i)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

NO_CHUNKS = 3
gnr_chnks = list(split(gnrs, NO_CHUNKS))

def kld_mp(chnk):
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

p = Pool(processes=3)

t1=time.time()
data = p.map(kld_mp, [i for i in gnr_chnks])
t2=time.time()

p.close()

# only 12% improvement with 4 instead of 3, not worth it i think
# makes me uneasy to see everything at 100%, wonder if its good for cpu

ar_lst = [np.array(i) for i in data]
ar_cb = np.concatenate(ar_lst, axis=0)

plt.hist(ar_cb[np.where(ar_cb < math.inf)], bins='auto')
plt.show()

# need to deal with inf values
# where do they come from? 0s in A i think
# assess badness of fit: sum of cells of a that are 0 in b should not be above X (0.95)

# *** test how much thrshold has to be relaxed to get most/all genres included
# quite alot; to the extent that most genres will have at least dozes of superordiates and suporidates
# -> how to 

gnr_cnt = []

for i in np.arange(0.01, 0.25, 0.0025):
    kld_el = sbst_eler(ar_cb, operator.lt, i)

    g_kld = Graph()
    g_kld_id = g_kld.add_edge_list(kld_el, hashed=True, string_vals=True)

    # print(g_kld)
    # print(i)

    gnr_cnt.append(len(list(g_kld.vertices())))

xs = np.arange(0.01, 0.25, 0.0025)
fig = plt.figure()
ax = plt.axes()
ax.plot(xs, gnr_cnt)
plt.show()

# *** normal kld continue

kld_el = sbst_eler(ar_cb, operator.lt, 0.12)

# kld_rel = np.where(np.array(klds) < 0.05)
# kld_el = np.array(kld_cmps)[kld_rel[0]]


g_kld = Graph()
g_kld_id = g_kld.add_edge_list(kld_el, hashed=True, string_vals=True)

graph_pltr(g_kld, g_kld_id, 'acst_spc4.pdf')

x = set(g_kld_id[i] for i in g_kld.vertices())

vd_kld, vd_kld_rv = vd_fer(g_kld, g_kld_id)

[print(g_kld_id[i]) for i in g_kld.vertex(vd_kld['indie']).in_neighbors()]

graph_draw(g_kld, output='g_kld.pdf')
# amount of reciprocal relationships?
# not if i just get 3 most influential parents or so

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

kld2_el = kld_n_prnts(ar_cb, 4)

g_kld2 = Graph()
kld_sim = g_kld2.new_edge_property('double')
g_kld2_id = g_kld2.add_edge_list(kld2_el, hashed=True, string_vals=True, eprops = [kld_sim])

vd_kld2,vd_kld2_rv = vd_fer(g_kld2, g_kld2_id)

kld_sim_int = g_kld2.new_edge_property('int16_t')

for e in g_kld2.edges():
    kld_sim_int[e] = math.ceil(kld_sim[e]*100)

graph_pltr(g_kld2, g_kld2_id, 'acst_spc5.pdf', 1)

# ** KDL 1 exploration 

case:
gnr = 'Death Doom Metal'
gnr_nbrs = [g_kld_id[i] for i in list(g_kld.vertex(vd_kld['Death Doom Metal']).out_neighbors())]
gnr_id = gnrs.index(gnr)

klds = kld_mp([gnr])

klds2 = [i for i in klds[0] if i < 0.05]
plt.hist(klds2, bins='auto')
plt.show()

for k in gnr_nbrs:
    i_v = acst_mat[gnrs.index(gnr)]
    k_v = acst_mat[gnrs.index(k)]

    b_zeros = np.where(k_v==0)
    a_sum_b_zeros = sum(i_v[b_zeros])
    prop_missing = a_sum_b_zeros/sum(i_v)
            
    if prop_missing == 0:
        ent = round(entropy(i_v, k_v),3)
        print(gnr, k, ent, 'complete')
                
    elif prop_missing < 0.05:
                
        i_v2 = np.delete(i_v, b_zeros)
        k_v2 = np.delete(k_v, b_zeros)

        ent = round(entropy(i_v2, k_v2),3)
        print(gnr, k, ent, 'incomplete', prop_missing)

x = [i for i in range(0,120,1)]

fig = plt.figure()
ax = plt.axes()
# ax.plot(x[0:10], i_v[0:10])
ax.plot(x[0:10], k_v[0:10])
plt.show()

a1,a2,nsns = plt.hist(acst_gnr_dict[gnr]['dncblt'], bins=10)
plt.hist(acst_gnr_dict[k]['dncblt'], bins=10)
plt.show()

fig = plt.figure()
ax = plt.axes()
ax.plot(x[0:10], a1/sum(a1))
# ax.plot(x[0:10], i_v[0:10])
plt.show()




# ax.plot(x, i_v)




# * feature extraction
# ** informativeness

# average of similarities of indegrees
# superordinates

gnr = 'dark ambient'
gnr = 'rock'

# spr_ord = list(g_kld2.vertex(vd_kld2[gnr]).in_neighbors())

# sim_vlu[ghrac.edge(gv, v)]
# sim_vlu[ghrac.edge(v, gv)]

v = spr_ord[0]

res_dict = {}
for gnr in gnrs:
    res_dict[gnr] = {}

    
for gnr in gnrs:
    # generate a whole bunch of measures
    
    gv = g_kld2.vertex(vd_kld2[gnr])

    # get sum of 3 distance to 3 parents
    prnt3_inf = gv.in_degree(kld_sim)
    res_dict[gnr]['prnt3_inf'] = prnt3_inf

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

    res_dict[gnr]['cohrt_pct_inf'] = cohrt_pct_inf
    res_dict[gnr]['cohrt_mean_non_inf'] = cohrt_mean_non_inf

df_res = pd.DataFrame(res_dict).transpose()

# *** debug w_std2: should not result in symmetric similarities
# similarites are symmetric, but have to be processed

def asym_sim(gnrs, vd):
    """generates asym mat based on weights scaled by max, making more equally distributed genres more general"""
    cmps = all_cmps_crubgs(gnrs, vd, 'product')

    all_sims = vertex_similarity(gac, 'dice', vertex_pairs = cmps, eweight = w_std2)

    sims_rows = np.split(all_sims, len(gnrs))
    sims_ar = np.array(sims_rows)

    deg_vec = [gac.vertex(vd[i]).out_degree(weight=w_std2) for i in gnrs]

    # equivalent to adding the two outdegrees together 
    deg_ar = np.array([deg_vec]*len(gnrs))
    deg_ar2 = (deg_ar + np.array([deg_vec]*len(gnrs)).transpose())/2

    # see how much is actually in common, equivalent of multiplication with similarity
    cmn_ar = deg_ar2*sims_ar

    # see the percentage of what is in common for each genre
    ovlp_ar = cmn_ar/deg_ar
    return(ovlp_ar)

asym_sim_ar = asym_sim(gnrs, vd)

el_asym = sbst_eler(asym_sim_ar, 0.99)
# this seems to put super general things (10 of 10 stars) as super wide
# wonder if i should use the reverse distance (superordinate to subordinate) somehow, might have relevant information

g_asym = Graph()
asym_weit = g_asym.new_edge_property('double')

asym_id = g_asym.add_edge_list(el_asym, hashed=True, string_vals=True, eprops = [asym_weit])

vd_asym, vd_asym_rv = vd_fer(g_asym, asym_id)

graph_pltr(g_asym, asym_id, 'acst_spc4.pdf')

for v in g_asym.vertices():
    print(asym_id[v], v.in_degree())

# x = np.reshape(asym_sim_ar, (1, len(gnrs)**2))
# plt.hist(x[0], bins='auto')
# plt.show()


# you imbecil
# you fucking moron
# when you get directed similarity you have to divide it by the bases
# first get common stuff
# then divide that by each vertex' ttl 

# * old stuff, maybe recyclable
# ** calculating overlap/divergence

# need to find cells where c2 is nonzero and c1 is zero

emp_cells = np.logical_and(c2.spc_fl > 0, c1.spc_fl== 0).nonzero()

emp_rows = [[0 for i in range(100)] for i in range(100)]
emp_spc = np.array(emp_rows)

for i in range(len(emp_cells[1])):
    emp_spc[emp_cells[0][i],emp_cells[1][i]]=c2.spc_fl[emp_cells[0][i],emp_cells[1][i]]

plt.imshow(emp_spc, interpolation='nearest')
plt.show()

pct_ncvrd = sum(sum(emp_spc))/sum(sum(c2.spc_fl))

c2_spc = c2.spc_fl

for i in range(len(emp_cells[1])):
    c2_spc[emp_cells[0][i], emp_cells[1][i]] = 0


xx = list(itertools.chain.from_iterable(c2_spc))
yy = list(itertools.chain.from_iterable(c1.spc_fl))

entropy(xx, yy)

yy2 = c1.h1 + c1.h2
xx2 = c2.h1 + c2.h2

entropy(xx2, yy2)


xs2 = [i for i in range(len(yy2))]
ax = plt.axes()
ax.plot(xs2, xx2)
ax.plot(xs2, yy2)
plt.show()


# ** kullback-Leibler divergence

x = [i for i in range(0,10)]
x1 =  [0,  0,0,0, 0.05,0.1,0.3,0.6,0.3,0.1]
x1_nz = [0.01,  0.01,0.01,0.01, 0.05,0.1,0.3,0.6,0.3,0.1]

x1n = [i/sum(x1) for i in x1]
x1_nzn = [i/sum(x1_nz) for i in x1_nz]

x1_sml= [i/2 for i in x1_nz]

x2 = [0.1,0.2, 0.3, 0.3, 0.2, 0.1, 0.1,0,0,0]
x2_nz = [0.1,0.2, 0.3, 0.3, 0.2, 0.1, 0.1,0.01,0.01,0.01]

x2n = [i/sum(x2) for i in x2]
x2_nzn = [i/sum(x2_nz) for i in x2_nz]


import matplotlib.pyplot as plt

ax = plt.axes()
ax.plot(x, x1_nz)
ax.plot(x, x1_sml)
plt.show()


def klbk_lblr_dist(x1,x2):
    dist = 0
    for i in range(0, len(x1)):
        if x2[i]!= 0 and x1[i] !=0:
            sub_res = x1[i] * math.log(x1[i]/x2[i])
            print(i, sub_res)
            dist = dist + sub_res
    print('-----------------')
    print(dist)

# doesn't work if either is 0: either fraction undefined, or log

klbk_lblr_dist(x1,x2)
klbk_lblr_dist(x2,x1)

klbk_lblr_dist(x1,x1_sml)
klbk_lblr_dist(x1_sml, x1)

# no zeroes
# Hannan use binary dimensions: just one value summarizes each dimension
# but using continuous doesn't seem to be qualitative different, just a bunch of more numbers to compare


entropy(x1, x2)
entropy(x2, x1)

klbk_lblr_dist(x1_nrml, x2_nrml)
klbk_lblr_dist(x2_nrml, x1_nrml)




x1_n = [i/sum(x1) for i in x1]
x2_n = [i/sum(x2) for i in x2]

entropy(x1_nrml, x2_nrml)
entropy(x2_nrml, x1_nrml)

entropy(x1_nz, x2_nz)

sum([x2_n[i] * math.log((x2_n[i] / x1_n[i])) for i in range(10)])
sum([x1_n[i] * math.log((x1_n[i] / x2_n[i])) for i in range(10)])

# works nice
need to compare with 0s and without

klbk_lblr_dist(x1n,x2n)
klbk_lblr_dist(x1_nzn,x2_nzn)
entropy(x1_nzn,x2_nzn)


klbk_lblr_dist(x2,x1)
klbk_lblr_dist(x2_nz,x1_nz)


x1_sz = [0, 0.05, 0.1, 0.2, 0.3,0.35, 0.3, 0.2, 0.1, 0.05]
x2_sz = [0, 0   , 0 , 0.1, 0.3,0.45, 0.3, 0.1, 0, 0]

x1_szn = [i/sum(x1_sz) for i in x1_sz]
x2_szn = [i/sum(x2_sz) for i in x2_sz]

entropy(x2_szn, x1_szn)

ax = plt.axes()
ax.plot(x, x1_szn)
ax.plot(x, x2_szn)
plt.show()

# ** hausdorff distance

from scipy.spatial.distance import directed_hausdorff

u = np.array([(1.0, 0.0),
              (0.0, 1.0),
              (-1.0, 0.0),
              (0.0, -1.0)])
u1 = [i[0] for i in u]


v = np.array([(2.0, 0.0),
              (0.0, 2.0),
              (-2.0, 0.0),
              (0.0, -4.0)])

plt.scatter(u[:,0], u[:,1])
plt.scatter(v[:,0], v[:,1])
plt.show()

directed_hausdorff(u,v)
directed_hausdorff(v,u)

HD is largest of shortest distances?
for each point get the shortest distance to other set, and then take the largest of those? 
seems to be

i think asymmetry is fine since unit of analysis is genre

seems to be sensititve to outliers tho (wikipedia figure)
https://en.wikipedia.org/wiki/Hausdorff_distance

PIazzai use mean of min minimum distances



# complete not the same:
# low fractions of x2 (0.01), if x1 fraction is high (0.2):
# overall fraction is 20
# log is not that high but still very high
# does not happen if i drop it

# KLD only works if for all x where x2 (qx) is 0, x1 (px) is also 0
# only works for sub-concept relation
# absolute continuity: some calculus stuff


# can KLD account for unequal likelihoods?
# does sum of 1 pose problem?
# at all levels, X more likely than Y
# seems to ask: given you have a object of distribution X, you likely is it to be in place Z
# not: given that you're in place Z, how likely are you to be part of X or Y?

# do i need it? could use it for sub-concept , if subset relation exists

# but would mean i need another measure for non subsets

# maybe makes sense: asking how similar are swimmers to athlethes is a different question than asking how similar are scientists to athletes

# Hannan also want to use KDV for cohort distinctiveness
# requires non-zero values on all features
# idk
# should not work if subconcepts are more specific, have not all the same dimensions
# swimmer (subconcept of athlete) has different dimensions than bodybuilder (lifts weight)
# probability distributions in those two dimensions are not overlapping

# Hannan use cosine similarity (p.91), then distances (exponential) 
# but that's symmetric
# maybe that bad in cohort tho: need to see volume distribution in cohorts
# also problem that sub-sub concepts (lowest level) will show up as members of higher level
# exclude if (next to being subconcept) it is also a subconcept of another subconcept

# cosine similarity: also doesn't take probability distribution into account
# first waste of information
# second 



klbk_lblr_dist(x2,x1)



# * scrap: basically useless
# ** cosine similarity
# might not even have to normalize for it, but won't really distort much me thinks

from sklearn.metrics.pairwise import cosine_similarity

x= cosine_similarity(acst_mat)


plt.hist(x[np.where(0<np.tril(x))], bins='auto')
plt.show()


plt.hist(x[(np.tril(x) > 0) & (np.tril(x) < 1)], bins='auto')
plt.show()

from scipy.spatial import distance
distance.euclidean

x2 = sklearn.metrics.pairwise.euclidean_distances(acst_mat)
plt.hist(x2[np.tril(x2) > 0], bins='auto')
plt.show()

# what's the point of putting it into network really
# -> need to functionalize the network generation
# but more the relevant feature extraction -> straightforward to compare


