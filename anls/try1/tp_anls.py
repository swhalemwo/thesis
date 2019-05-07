import json
import os
import csv

import numpy as np
import graph_tool as gt
from graph_tool.all import *
from graph_tool import *  


import sqlite3
from os import listdir




# todos

## need general analysis function 
## also general output function


# done
# write graphs to gt objects





def get_mbid_v_dict():

    vcnt = len(list(gx.vertices()))

    cntr = 0
    cntr2 = 0

    mbid_v_dict = {}
    for i in range(0,vcnt):
        mbid_v_dict[gx.vp.name[i]] = i

        cntr+=1
        cntr2 +=1
        if cntr2 == 5000:
            print(cntr, round(cntr/vcnt,2))
            cntr2 = 0

    return(mbid_v_dict)




def get_mbids(min_in_deg):
    rel_albs = find_vertex_range(gx, 'in', (min_in_deg, 10**10))
    rel_mbids = [gx.vp.name[i] for i in rel_albs]
    return(rel_mbids)


def sqlite_setup():
    tag_sqldb="/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_tags1.sqlite"
    conn = sqlite3.connect(tag_sqldb)
    c=conn.cursor()
    return(c)

# rel_mbids_sub = rel_mbids[0:10]


# sql = "select mbid, tag, weight from tags where mbid in ({seq})".format(
#     seq=','.join(['?']*len(rel_mbids_sub)))

# stags = c.execute(sql, rel_mbids_sub).fetchall()

def get_unq_tags(rel_mbids, min_weight, min_cnt):

    sql2 = "select distinct tag from tags where weight > " + str(min_weight) + " and mbid in ({seq})".format(
        seq=','.join(['?']*len(rel_mbids)))
    
    sql2 = sql2 + " group by tag having count (*) > " + str(min_cnt)

    unq_tags = c.execute(sql2, rel_mbids).fetchall()
    unq_tags = [i[0] for i in unq_tags]
    return(unq_tags)


# ut_dict = {}
# for i in unq_tags:
#     ut_dict[i[0]] = 0


def get_tag_mbid_dict(rel_mbids, unq_tags, min_weight):
    sql = "select mbid, tag, weight from tags where weight > 10 and mbid in ({seq})".format(
        seq=','.join(['?']*len(rel_mbids)))

    sql = sql + " and tag in ({seq})".format(
        seq=','.join(['?']*len(unq_tags)))

    daet_ttl = c.execute(sql, rel_mbids+unq_tags).fetchall()

    tag_mbid_dict ={}

    for i in daet_ttl:
        if i[1] in tag_mbid_dict.keys():
            tag_mbid_dict[i[1]].append((i[0], i[2]))
        else:
            tag_mbid_dict[i[1]] = [(i[0], i[2])]

    return(tag_mbid_dict)



def get_weighted_plcnt(utx):

    tag_rel_mbids = [i[0] for i in tag_mbid_dict[utx]]
    tag_rel_weits = [i[1]/100 for i in tag_mbid_dict[utx]]

    tag_rel_v_ids = [mbid_v_dict[i] for i in tag_rel_mbids]
    tag_rel_v_dgs = [gx.vertex(i).in_degree() for i in tag_rel_v_ids]

    tag_plcnt = sum([a*b for a,b in zip(tag_rel_weits,tag_rel_v_dgs)])

    return(tag_plcnt)


def res_wr(t, tag, res, outfile):
    res_str = [str(i) for i in res]
    prntrow = [t, tag] + res_str
    
    with open(outfile, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow(prntrow)



gt_dir = "/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/stage/gts/"
gt_files = listdir(gt_dir)


for i in gt_files:
    print('load graph')
    tp = i[0:21]
    gx=load_graph(gt_dir + i)

    outfile= "/home/johannes/Dropbox/gsss/thesis/anls/try1/results/t1.csv"    

    print('creating mbid vertex index')
    mbid_v_dict = get_mbid_v_dict()

    print('get relevant mbids')
    rel_mbids = get_mbids(150)
    c = sqlite_setup()

    print('get relevant tags')
    unq_tags = get_unq_tags(rel_mbids, 10, 10)

    print('build tag mbid dict')
    tag_mbid_dict = get_tag_mbid_dict(rel_mbids, unq_tags, 10)
    
    print ('analyze')
    for utx in unq_tags:
        tag_plcnt = get_weighted_plcnt(utx)
        res = [tag_plcnt]

        res_wr(tp, utx, res, outfile)

        print(utx, tag_plcnt)




## need dict of mbids to vertex index
## vertex count: where was it? what was it important for? density, but apparently not



# virtually all time goes into sql call
# just hope it doesn't scale too bad properly
# not retrieving tags doesn't improve really anything




# differet number of tags because i can't really filter out the tags i don't want?
# could add requirement of being in unq tags
# then a bunch of tags are missing
# comes from overly permissive unique tag call: should be limited to mbids


# diffx = set(unq_tags2) - set(tag_mbid_dict.keys())



# unq_tags2 = [i[0] for i in unq_tags]

# daet_ttl = c.execute(sql, rel_mbids+unq_tags2).fetchall()
# # sql requests seem to be taking almost the same time regardless of size?

# mbid_tag_dict = {}
# cntr =0
# cntr2 = 0

# for i in daet_ttl:
#     if i[0] in mbid_tag_dict.keys():
#         mbid_tag_dict[i[0]].append((i[1], i[2]))
#     else:
#         mbid_tag_dict[i[0]] = [(i[1], i[2])]

#     cntr+=1
#     cntr2 +=1
#     if cntr2 == 1000:
#         print(cntr)
#         cntr2 = 0


# ## need other way you moron


