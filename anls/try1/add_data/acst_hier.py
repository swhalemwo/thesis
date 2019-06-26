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

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


client = Client(host='localhost', password='anudora', database='frrl')

avbl_vars = ['id', 'dncblt', 'gender', 'timb_brt', 'tonal', 'voice', 'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy', 'mood_party', 'mood_relaxed', 'mood_sad', 'gnr_dm_alternative', 'gnr_dm_blues', 'gnr_dm_electronic', 'gnr_dm_folkcountry', 'gnr_dm_funksoulrnb', 'gnr_dm_jazz', 'gnr_dm_pop', 'gnr_dm_raphiphop', 'gnr_dm_rock', 'gnr_rm_cla', 'gnr_rm_dan', 'gnr_rm_hip', 'gnr_rm_jaz', 'gnr_rm_pop', 'gnr_rm_rhy', 'gnr_rm_roc', 'gnr_rm_spe', 'gnr_tza_blu', 'gnr_tza_cla', 'gnr_tza_cou', 'gnr_tza_dis', 'gnr_tza_hip', 'gnr_tza_jaz', 'gnr_tza_met', 'gnr_tza_pop', 'gnr_tza_reg', 'gnr_tza_roc', 'mirex_Cluster1', 'mirex_Cluster2', 'mirex_Cluster3', 'mirex_Cluster4', 'mirex_Cluster5', 'length', 'label', 'lang', 'rl_type', 'rls_cri']

chsn_vars = ['dncblt', 'gender', 'timb_brt', 'tonal', 'voice']

ids = [avbl_vars.index(i) for i in chsn_vars]


rows_acst = client.execute("""select lfm_id, dncblt,gender,timb_brt,tonal,voice from acstb 
join (select mbid as lfm_id, cnt from song_info3 where cnt > 400 ) using lfm_id""")

df_acst = pd.DataFrame(rows_acst, columns = ['lfm_id', 'dncblt','gender', 'timb_brt','tonal', 'voice'])

songs_acst = df_acst['lfm_id']


# can make the query longer to select only genres which occur at least X times
# join statements have to be on different levels i think, that's why i have to make three new lines to take into account the tag appearnace (tag_ap )

# have to put another acstb condition in there, which is more restrictive
# hm now have tags for less than i have acoustic information
# which makes sense because some for which i have acoustic information fail to meet tag requirements
# hm could adjust the query
# idk maybe is easier to pop all that are in acst_df but not in tag_df? 

rows_tags = client.execute("""select mbid, tag, rel_weight, cnt from
(select * from 
    (select * from tag_sums
        join (select tag, count(*) as tag_ap from tag_sums where weight > 10 and rel_weight > 0.05
            group by tag having tag_ap > 10) using tag)
    join (select lfm_id as mbid from acstb) using mbid)
join (select mbid, cnt from song_info3 where cnt > 400 ) using mbid  
where weight > 10 and rel_weight > 0.05""")

df_tags = pd.DataFrame(rows_tags, columns=['lfm_id', 'tag', 'rel_weight', 'cnt'])

songs_tags = df_tags['lfm_id']
len(np.unique(songs_tags))

# select songs that have to be deleted
songs_tbp = list(set(songs_acst) - set(songs_tags))

# make lfm_id, row_id dict
df_acst_dict = {}
c = 0
for r in df_acst.itertuples():
    df_acst_dict[r.lfm_id] = c
    c +=1

# rows to pop
rstp = []
for i in songs_tbp:
    # shouldn't every songs_tbp in df_acst???
    # yup: songs_tbd are those that are in df_acst but not in df_tags
    # which means they are in df_acst
    rtp = df_acst_dict[i]
    rstp.append(rtp)

df_acst2 = df_acst.drop(rstp)
# finally works

# make position dicts because dicts good

gnr_song_dict = {}
for r in df_tags.itertuples():
    gnr = r.tag
    
    if gnr in gnr_song_dict.keys():
        gnr_song_dict[gnr].append(r.lfm_id)
    else:
        gnr_song_dict[gnr] = [r.lfm_id]
        

acst_pos_dict = {}
for r in df_acst2.itertuples():
    acst_pos_dict[r.lfm_id] = r.Index


unq_tags = list(np.unique(df_tags['tag']))
# for i unq_tags:

gnr = 'black metal'
gnr_acst_ids = [acst_pos_dict[i] for i in gnr_song_dict[gnr]]

df_gnr_tags = df_tags[df_tags['tag']==gnr]
df_gnr_acst = df_acst.loc[gnr_acst_ids]

df_gnr_cbmd = pd.merge(df_gnr_tags, df_gnr_acst, on='lfm_id')

weighted_avg_and_std(df_gnr_cbmd['dncblt'], df_gnr_cbmd['cnt'])

bins = np.arange(0, 1, 0.1)
a1, a0 = np.histogram(df_gnr_cbmd['dncblt'], bins=bins)

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


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)


y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(3, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(3, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.subplot(3, 1, 3)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')


def sb_pltr(ttl, row, nbr, xs, ys, xlbl, ylbl):
    plt.subplot(ttl, row, nbr)
    plt.plot(xs, ys, '.-')
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)

c = 0
for v in chsn_vars:

    # wrap into genre function
    gnr_acst_ids = [acst_pos_dict[i] for i in gnr_song_dict[gnr]]

    df_gnr_tags = df_tags[df_tags['tag']==gnr]
    df_gnr_acst = df_acst.loc[gnr_acst_ids]

    df_gnr_cbmd = pd.merge(df_gnr_tags, df_gnr_acst, on='lfm_id')
    bins = np.arange(0, 1, 0.1)
    a1, a0 = np.histogram(df_gnr_cbmd[v], bins=bins)

    ttl = 5
    c +=1
    sb_pltr(ttl, 1, c, ax_lbl, a1, 'freq', v)

plt.show()


# g1 = 'black metal'
# g2 = 'metal'


# 371 comparisons in per sec
# 16 mil is gonna take like 12 hours
# and it's not even weighted yet
# 




db = DiscoDB(gnr_song_dict)

# t1 = time.time()
# for i in range(100):
#     x = (list(db.query(Q.parse('rock & pop rock'))))
# t2 = time.time()    
# 806/sec, not bad
# performance goes a bit down (738-780) when running multiple processes, oh well, NBD


# t1 = time.time()
# for i in range(500):

#     g1_entrs = gnr_song_dict[g1]
#     g2_entrs = gnr_song_dict[g2]

#     x = set.intersection(set(g1_entrs), set(g2_entrs))
#     # print(len(x)/len(g1_entrs))

# t2 = time.time()

def gnr_sub_disco(g1, g2):
    """compare overlap between two genres
    outspeeded by metaqueries"""
    qry1 = g1 + " & " + g2
    ovlp = len(list(db.query(Q.parse(qry1))))
    ovlp_prop = ovlp/len(gnr_song_dict[g1])
    return(ovlp_prop)





# [x for _,x in sorted(zip(Y,X))]

tag_cnts = [len(gnr_song_dict[i]) for i in unq_tags]

# get tags sorted by playcount or appearance or something
tags_srtd=[x for _,x in sorted(zip(tag_cnts, unq_tags), reverse=True)]



# ovlp_res = []

# for g1 in tags_srt_sub:
#     g1_ovlps = []
#     for g2 in tags_srt_sub:
#         ovlp = gnr_sub_disco(g1, g2)
#         g1_ovlps.append(ovlp)

#     ovlp_res.append(g1_ovlps)


# ovlp_ar = np.array(ovlp_res)


# a = np.histogram(ovlp_ar, bins=30)

# a0 = np.append(np.array([0]), a[0])
# a1 = [i*10 for i in a[1]]


# artists seem to be between clusters: rammstein, marilyn manson, korn, disturbed, system of a down, linkin park
# would suck to exclude them, they've basically become a brand now
# are part of the cognitive web that people use to classify music



# plt.bar(a1, a0)
# plt.show()


# plt.hist(a, bins='auto')
# plt.show()

# just text no vertex so much nicer
# graphviz dot good: can handle linebreaked titles


# can add transitivity violations as variable:
# see if any there are supersets of supersets that are not supersets of X
    
# fuck yeah
# could also be way of filtering: only relevant genres are those that either have subgenres or are subgenres of something

# see if replication of values at lower levels:




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


# df = pd.DataFrame(columns = ['id'] + chsn_vars)
# df[0] = row[0] + float(row[1]) + float(row[2]) + float(row[3]) + float(row[4]) + float(row[5])

# df.loc[0] = entry

# row_dict = {'id':row[0], 'dncblt':float(row[1]), 'gender':float(row[2]), 'timb_brt':float(row[3]), 'tonal':float(row[4]), 'voice':float(row[5])}

# df = pd.DataFrame([row_dict])



# entry = [row[0]] + [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]

# df = pd.DataFrame(columns = ['id'] + chsn_vars)
# dims_file = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/acstbrnz.csv'
# c = 0
# row_list = []
# with open(dims_file, 'r') as fi:
#     rdr = csv.reader(fi)

#     for row in rdr:

#         row_dict = {'id':row[0], 'dncblt':float(row[1]), 'gender':float(row[2]), 'timb_brt':float(row[3]), 'tonal':float(row[4]), 'voice':float(row[5])}
#         row_list.append(row_dict)

#         c+=1
        
#         # entry = [row[0]] + [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
#         # df.loc[c] = entry
        
#         # df[c] = row[0:6]
# dk1 = 0
# for i in range(0,10):
#     if x2[i] !=0 and x1[i] != 0:
#         sub_res = x1[i] * np.log(x1[i]/x2[i])
#         dk1 = dk1 + sub_res
#         print(sub_res)
#     else:
#         print('lel')

# * older ch queries
# rows_tags = client.execute("""select mbid, tag, rel_weight, cnt from tag_sums 
# join (select mbid, cnt from song_info3 where cnt > 400 ) using mbid 
# where weight > 10 and rel_weight > 0.05""")


