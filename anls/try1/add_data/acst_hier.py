import csv
import json
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import math
import time
from discodb import DiscoDB, Q
import matplotlib.pyplot as plt

from scipy.stats import entropy
from sklearn import preprocessing

from graph_tool.all import *
from graph_tool import *

# * hierarchical relations based on split dimensions
# functions defined in item_tpclt

def vd_fer(g, idx):
    """creates vertex index dict from graph and id property map"""
    # vertex_dict
    vd = {}
    for i in g.vertices():
        vd[idx[i]] = int(i)
    return(vd)


el_ttl = []
ts = []
gnrs = ['metal', 'ambient', 'rock', 'hard rock', 'electronic', 'death metal']*10

for gnr in gnrs:
    # print(gnr)
    t1 = time.time()


    dfc = get_df_cbmd(gnr)
    gnr_cnt = len(dfc)
    el_gnr = []
    # print(len(el_gnr))
    t2 = time.time()

    for vrbl in vrbls:

        bins = np.arange(0, 1.1, 0.1)
        a1, a0 = np.histogram(dfc[vrbl], bins=10)

        nds1 = [vrbl + str(i) for i in range(1, len(bins))]
        nds2 = [gnr]*len(nds1)

        a_wtd = [i/gnr_cnt for i in a1]

        elx = [i for i in zip(nds2, nds1, a1, a_wtd)]

        el_gnr = el_gnr + elx
    t3 = time.time()

    el_ttl = el_ttl + el_gnr
    t4 = time.time()

    ts.append([t1,t2,t3,t4])

# not that fast, might be optimized
# atm takes especially long for long genres
# NOT GOOD
# can be parallelized but still
# iloc is some improvement, especially for longer ones? seems so
# think i can shave off some time by doing an additional dict for gnrs with gnrs as keys and acst_df as values
# would save queries..
# but for rock it's currently 0.02 of 0.12
# more substantial savings by throwing out pandas

# graph acoustic 
gac = Graph()

w = gac.new_edge_property('int16_t')
w_std = gac.new_edge_property('double')

idx = gac.add_edge_list(el_ttl, hashed=True, string_vals=True, eprops = [w, w_std])

vd = vd_fer(gac, idx)

# needs to be generalized into list for relevant pairs 
# especially to assess prevalence of non-binary fit and hence possibility of overlap misattribution (lack sub due to area not covered)

vertex_similarity(gac, 'dice', vertex_pairs = [(vd['ambient'],vd['electronic'])])
# WTF start
vertex_similarity(gac, 'dice', vertex_pairs = [(vd['ambient'],vd['electronic'])], eweight = w)
vertex_similarity(gac, 'jaccard', vertex_pairs = [(vd['electronic'],vd['ambient'])], eweight = w)
# WTFFF end: 

vertex_similarity(gac, 'dice', vertex_pairs = [(vd['ambient'],vd['electronic'])], eweight = w_std)
vertex_similarity(gac, 'dice', vertex_pairs = [(vd['ambient'],vd['death metal'])], eweight = w_std)


# [print(w[gac.edge(vd['ambient'], vd['voice' + str(i)])]) for i in range(1,11)]
# [print(w[gac.edge(vd['electronic'], vd['voice' + str(i)])]) for i in range(1,11)]


g.vertex(vd['electronic']).out_neighbors()

# * general comparison function

gnr_ids = [vd[i] for i in gnrs]

# needs size dict for direction
# might need to integrate some weights
sz_dict = {}
for gnr in gnrs:
    weit_ttl = sum([w[v] for v in gac.vertex(vd[gnr]).out_edges()])/5
    sz_dict[gnr] = weit_ttl

# order doesn't matter anymore due to standardization
# do i need a matrix?
# can read it back in (either lower triangle columnwise or upper rowise)
# not sure if needed tho
# can just the positions of those over threshold, get the corresponding original comparisions, get the corresponding genres, and add that to edge list

comps = list(itertools.combinations(gnr_ids, 2))

# nested loops to ensure direction
# smaller one is now first: relation to be tested is subsetness

cprx = []
for i in range(lenx):
    cprx2 = []
    for k in range(i,lenx):
        if i==k:
            next
        else:
            v1, v2 = gnrs[i], gnrs[k]
            v1_sz, v2_sz = sz_dict[v1], sz_dict[v2]

            if v1_sz > v2_sz:
                cprsn = [v2, v1]
            else:
                cprsn = [v1, v2]

            cprx2.append(cprsn)
    if len(cprx2) > 0:
        cprx.append(cprx2)

cmps = list(itertools.chain.from_iterable(cprx))


# * some old stuff

# ** plotting dists for multiple varibales and genres
fig, ax = plt.subplots()
# ax.bar(a0[:-1], a1)

ax.plot(a1)
plt.show()

ax_lbl = []
for i in range(len(a1)):
    print(i)
    ax_lbl.append(str(round(a0[i],2)) + "-" + str(round(a0[i+1],2)))

plt.bar(ax_lbl, a1)
plt.show()



def sb_pltr(ttl, row, nbr, xs, ys, xlbl, ylbl):
    plt.subplot(ttl, row, nbr)
    plt.plot(xs, ys, '.-')
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)



gnrs = ['rap', 'metal', 'ambient', 'hard rock', 'pop', 'punk', 'ska']


for gnr in gnrs: 
    c = 0
    for v in vrbls:

        # wrap into genre function
        gnr_acst_ids = [acst_pos_dict[i] for i in gnr_song_dict[gnr]]

        df_gnr_tags = df_tags[df_tags['tag']==gnr]
        df_gnr_acst = df_acst.loc[gnr_acst_ids]

        df_gnr_cbmd = pd.merge(df_gnr_tags, df_gnr_acst, on='lfm_id')
        bins = np.arange(0, 1, 0.1)
        a1, a0 = np.histogram(df_gnr_cbmd[v], bins=bins, weights=df_gnr_cbmd['rel_weight'])

        ttl = 5
        c +=1
        sb_pltr(ttl, 1, c, ax_lbl, a1, 'freq', v)

plt.show()
# i don't like this at all
# maybe i shouldn't treat variables as continuous
# rather binary and weigh?
# idk.. would mean separate weights for each cell

# think i should functionify the df creation
# use item_tpclt_for it


# multiple lines: i guess i should pass dimensions there earlier

 
# g1 = 'black metal'
# g2 = 'metal'

# 371 comparisons in per sec
# 16 mil is gonna take like 12 hours
# and it's not even weighted yet
# 






# * infering probability distributions from vectors

import matplotlib.pyplot as plt
import math

class dist_cpr:
    """generates 2D probability distribution given raw data (set has separate parameter for each dimension)"""
    
    def summr(self, s, min_s, max_s):
        """creates hists of the distributions"""
        sums1 = []
        for i1 in np.arange(min_s, max_s, res):
            i2 = i1 + res
            buckets = []
            buckets = [x for x in s if x > i1 and x < i2]
            sums1.append(len(buckets))

        return(sums1)        

    def spc_mlt(self, v1,v2):
        """re-calculate (multiply) 2d space from histogram info"""
        space = []
        for i in v1:
            v1s = []
            for k in v2:
                v1s.append(i*k)
            space.append(v1s)

        spc_ar = np.array(space)
        
        return(spc_ar)
    

    def vx_cructr(self, s1,s2):
        """combines input to get single vector for easier processing"""
        c = 0
        vx = []
        for s1x in s1:
            s2x = s2[c]
            vx.append((s1x, s2x))
            c+=1
        return(vx)

    def spc_cructr(self, min_s, max_s, res):
        """constructs the space for prob dist, dict is faster for now (than looping over each entry)"""
        spc = {}
        for i in np.arange(min_s, max_s, res):
            i1 = math.ceil(i)
            # print(i)
            spc[i1] = {}

            for k in np.arange(min_s, max_s, res):
                k1 = math.ceil(k)
                spc[i1][k1] = 0

        return(spc)


    def spc_flr(self, vx, spc, min_s, max_s, res):
        """fill up the dict space, convert it into list of lists and then np array"""
        # fill space
        fails = 0
        for i in vx:
            x = math.floor(i[0]/res)*res
            y = math.floor(i[1]/res)*res
            try:
                spc[x][y]+=1
            except:
                fails+=1
                pass
        print(fails)

        lls = []
        for i in np.arange(min_s, max_s, res):
            i1 = math.ceil(i)
            ll = []

            for k in np.arange(min_s, max_s, res):
                k1 = math.ceil(k)
                ll.append(spc[i1][k1])

            lls.append(ll)

        spc_fl = np.array(lls)
        return(spc_fl)

    
    def __init__(self, xs, ys, res, min_s, max_s):
        self.h1 = self.summr(xs, min_s, max_s)
        self.h2 = self.summr(ys, min_s, max_s)

        self.spc_ar = self.spc_mlt(self.h1, self.h2)
        self.vx = self.vx_cructr(xs, ys)
        print(len(self.vx))
        self.spc = self.spc_cructr(min_s, max_s, res)
        print(len(self.spc))

        self.spc_fl = self.spc_flr(self.vx, self.spc, min_s, max_s, res)




mu1, sigma1 = 50, 20
mu2, sigma2 = 40, 15
s1 = np.random.normal(mu1, sigma1, 40000)
s2 = np.random.normal(mu2, sigma2, 40000)
res = 5

min_s = 0
max_s = 100

c1 = dist_cpr(s1, s2, res, min_s, max_s)

xs = [i for i in np.arange(max_s/res)]


plt.imshow(c1.spc_ar, interpolation='nearest')
plt.show()

plt.imshow(c1.spc_fl, interpolation='nearest')
plt.show()


s3 = np.random.normal(30, 8, 10000)
s4 = np.random.normal(25, 5, 10000) 

c2 = dist_cpr(s3, s4, res, min_s, max_s)

cmb_ar = c1.spc_fl + c2.spc_fl

# plt.imshow(c1.spc_fl, interpolation='nearest')
plt.imshow(cmb_ar, interpolation='nearest')
plt.show()

ax = plt.axes()
ax.plot(xs, c1.h1)
ax.plot(xs, c1.h2)

ax.plot(xs, c2.h1)
ax.plot(xs, c2.h2)

plt.show()

# * calculating overlap/divergence

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



# * kullback-Leibler divergence

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

# * hausdorff distance

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


# * scrap 



