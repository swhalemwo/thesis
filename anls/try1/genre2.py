import csv
import json
import os
import urllib.request
from sklearn.manifold import MDS
from collections import Counter
import cairo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from graph_tool.all import *
from graph_tool import *  
import timeit
import time
import datetime
from datetime import datetime

csv.field_size_limit(100000000)

# g = Graph()
# act_id=g.new_vertex_property("string")
# lsn_dt=g.new_edge_property("int")

edges = []
lsn_dts = []
with open('/home/johannes/mega/gsss/lastfm-dataset-1K/8m_cmbd.csv', 'r') as fi:
    rdr = csv.reader(fi, delimiter = " ")
    cntr = 0

    for row in rdr:

        # 4466960

        try:
            dt=int(row[2])
            edges.append((row[0], row[1]))
            # edges.append((row[0], row[1], row[2]))
            lsn_dts.append(dt)
            # print(row)
        except:
            pass

        # cntr +=1
        # if cntr ==10:
        #     break


# g.add_edge_list([(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])

# edges=[(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]


# edges2 = [('1', '2'),
#          ('1', '3'),
#          ('2', '4')]




t1 = time.time()
g = Graph()
lsn_dt=g.new_edge_property("int")

## field: connections to all other fields?
## if fields A and B are connected, does failure in field b increase probability of failure in field b? 
## field takes forever tho



some_map = g.add_edge_list(edges, string_vals=True, hashed=True, eprops=lsn_dt)

g.add_edge_list(edges, string_vals=True, hashed=True, eprops=lsn_dt)
t2 = time.time()
t2 - t1

# read in time periods
lsn_dt.a = lsn_dts


# JFC is this fast, 4 seconds for a million entries
# 
subs = find_vertex_range(g, 'in', (100, 100000000))

NART = len(subs)

sims=np.empty([0,NART])
for k in subs[0:NART]:
    itrbl=[]
    for l in subs[0:NART]:
        itrbl.append((int(k),int(l)))

    # sim = graph_tool.topology.vertex_similarity(g, 'jaccard', vertex_pairs=itrbl).tolist()
    sim = graph_tool.topology.vertex_similarity(GraphView(g, reversed=True), 'dice', vertex_pairs=itrbl).tolist()
    # sim = graph_tool.topology.vertex_similarity(GraphView(g, reversed=True), 'jaccard', vertex_pairs=itrbl).tolist()    
    sims = np.append(sims, [sim], axis=0)

for i in range(NART): 
    for k in range(NART): 
        v_min = min(sims[i,k],sims[k,i])
        sims[k,i] = v_min
        sims[i,k] = v_min

dsims = 1-sims

model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(dsims)
plt.scatter(out[:, 0], out[:, 1])
# plt.axis('equal');
plt.show()


a = plt.hist(dsims, normed=True, bins=30)
plt.show()

np.histogram(dsims)


## * test similarity
nvs



# make list of artists for each user
# update edge

stuff = [i for i in v.out_neighbors()]
from collections import Counter

stuff_c = Counter(stuff)

stuff_ch = [i for i in stuff_c[0:10]]

words = Counter(f.read().split()).most_common(10)



# * plotting dissimilarities 

a = np.histogram(dsims)

a0 = np.append(np.array([0]), a[0])
a1 = a[1]

plt.bar(a1, a0)
plt.show()


dsimsll=[]

for i in dsims:
    for x in i:
        dsimsll.append(x)

plt.hist(dsimsll, bins=30)
plt.show()

# looks like way too much dissimilarity; most seem completely (!) unrelated to each other
# might be that my approach forces stuff from different time periods together..

# also, how to compare similarity across time periods?
# clusters: based on useres: each cluster has a measure how much of a user is in it -> 


# * cut by time

plt.hist(lsn_dts, bins=20)
plt.show()

# 1.14, 1.16, 1.18, 1.20, 1.22, 124


ts = 1140000000
ts = 1145000000

print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))


t1_list = []
for i in lsn_dts:
    if i > 1140000000 and i < 1145000000:
        t1_list.append(True)
    else:
        t1_list.append(False)


t1_filt = g.new_edge_property('bool')
t1_filt.a = t1_list

# edges_t1 = find_edge_range(g, lsn_dt, (1140000000, 1160000000))
g_t1 = GraphView(g, efilt=t1_filt)



subs1 = find_vertex_range(g_t1, 'in', (40, 100000000))

t1_filt_dgr = g.new_vertex_property('bool')

for i in subs:
    t1_filt_dgr[i] = True

    for k in i.in_neighbors():
        t1_filt_dgr[k]=True
    

gt1x = GraphView(g_t1, vfilt=t1_filt_dgr, reversed=True)
# actually makes sense, there are only 1k people in total, so just 300 here is not that strange tbh

# not clear which graph to use for the eventual subset
# there shouldn't be much of a difference but somehow there is
subs2 = find_vertex_range(gt1x, 'in', (100, 100000000))
subs3 = find_vertex_range(g, 'in', (2000, 100000000))

subs2 = subs3

NART = len(subs2)

gt1x = g

sims=np.empty([0,NART])
for k in subs2[0:NART]:
    itrbl=[]
    for l in subs2[0:NART]:
        itrbl.append((int(k),int(l)))

    # sim = graph_tool.topology.vertex_similarity(g, 'jaccard', vertex_pairs=itrbl).tolist()
    # sim = graph_tool.topology.vertex_similarity(gt1x, 'dice', vertex_pairs=itrbl).tolist()
    sim = graph_tool.topology.vertex_similarity(GraphView(gt1x, reversed=True), 'jaccard', vertex_pairs=itrbl).tolist()    
    sims = np.append(sims, [sim], axis=0)
    print(sims.shape)

for i in range(NART): 
    for k in range(NART): 
        v_min = min(sims[i,k],sims[k,i])
        sims[k,i] = v_min
        sims[i,k] = v_min

dsims = 1-sims

dsimsll=[]

for i in dsims:
    for x in i:
        dsimsll.append(x)

plt.hist(dsimsll, bins=30)
plt.show()

# maybe approach is wrong: assumes that what people listen to is similar
# what if people listen to a lot of stuff different?
# but stuff is only different if boundaries exist in the first place


t1 = time.time()
model = MDS(n_components=10, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(dsims)
plt.scatter(out[:, 0], out[:, 1])
# plt.axis('equal');
plt.show()
t2 = time.time()




dsims_sqrd = sum(sum(dsims**2))
dsims_sqrd = (dsims**2).sum() / 2
# not clear if i have to divide by 2
# looks like this in online code
# would make sense if i want distances as such (each is included twice)
# compare in R 

stress1 = np.sqrt(model.stress_ / dsims_sqrd)


# needs to be automated, but not too difficult: can make number of seconds per year and add that to start date
# points are still scattered, but dissimilarities look better:
# also less concentration on edges, but might be lack of points
#


from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps = 0.4, min_samples=4, metric='precomputed', leaf_size=10).fit(dsims)
Counter(clustering.labels_)

from sklearn.cluster import DBSCAN
import numpy as np
X = np.array([[1, 2], [2, 2], [2, 3],
              [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)

# eh not good


g.save('/home/johannes/Dropbox/gsss/thesis/anls/try1/g8m.gt')


g = load_graph('/home/johannes/Dropbox/gsss/thesis/anls/try1/g8m.gt')

