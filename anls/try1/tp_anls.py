import json
import os

import numpy as np
import graph_tool as gt
from graph_tool.all import *
from graph_tool import *  



import sqlite3
from os import listdir




# todos
# write graphs to gt objects


stage_dir = "/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/stage/"
stage_files = listdir(stage_dir)

# gx=load_graph(stage_dir + 'gts/' + '1167606000-1175378400.gt')


# for i in stage_files[1:25]:
#     print(i)

#     gx = gt.load_graph_from_csv(stage_dir+i, directed=True, string_vals=True, csv_options={'delimiter':'\t'})

#     gx_name = stage_dir + 'gts/' + i[0:21] + ".gt"

#     gx.save(gx_name)

# internal vertex maps should be there, seem to 


# get albums at point in time with playcount > X
# get tags from sqldb
# get all tags with weights > Y


rel_albs = find_vertex_range(gx, 'in', (300, 10**10))
rel_mbids = [gx.vp.name[i] for i in rel_albs]


tag_sqldb="/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_tags1.sqlite"
conn = sqlite3.connect(tag_sqldb)
c=conn.cursor()

rel_mbids_sub = rel_mbids[0:10]


sql = "select mbid, tag, weight from tags where mbid in ({seq})".format(
    seq=','.join(['?']*len(rel_mbids_sub)))

stags = c.execute(sql, rel_mbids_sub).fetchall()

# sql2 = "select distinct tag from tags where weight > 10 and mbid in ({seq})".format(
#     seq=','.join(['?']*len(rel_mbids_sub)))

sql2 = "select distinct tag from tags where weight > 10 group by tag having count(*) > 100"

unq_tags = c.execute(sql2).fetchall()
ut_dict = {}
for i in unq_tags:
    ut_dict[i[0]] = 0

# need sql call for each tag i think

utx = unq_tags[0][0]

def get_weighted_plcnt(utx):

    # qry_lst = list(rel_mbids_sub)
    qry_lst = list(rel_mbids)
    qry_lst.append(utx)

    sql3 = "select mbid, weight from tags where weight > 10 and mbid in ({seq}) and tag=?".format(
        seq=','.join(['?']*len(rel_mbids)))

    sql3 = "select mbid, weight from tags where weight > 10 and tag=?"

    daet_utx = c.execute(sql3, (utx,)).fetchall()
    # something wrong with sql, maybe need to filter afterwards? 

    # tag weights
    tag_rel_mbids = [i[0] for i in daet_utx]
    tag_rel_weits = [i[1]/100 for i in daet_utx]
    
    tag_get_ids = [mbid_v_dict[i] for i in tag_rel_mbids]
    tag_rel_dgs = [gx.vertex(i).in_degree() for i in tag_get_ids]

    tag_plcnt = sum([a*b for a,b in zip(tag_rel_weits,tag_rel_dgs)])

    return(tag_plcnt)



# virtually all time goes into sql call
# just hope it doesn't scale too bad properly
# not retrieving tags doesn't improve really anything


for utx in unq_tags:
    print(utx, ii)
    ut_dict[utx[0]] = get_weighted_plcnt(utx[0])


ut_dict[utx] = tag_plcnt

## need dict of mbids to vertex index
## vertex count: where was it? what was it important for? density, but apparently not
##


vcnt = len(list(gx.vertices()))

cntr = 0
cntr2 = 0

mbid_v_dict = {}
for i in range(0,vcnt):
    mbid_v_dict[gx.vp.name[i]] = i

    cntr+=1
    cntr2 +=1
    if cntr2 == 1000:
        print(cntr)
        cntr2 = 0











