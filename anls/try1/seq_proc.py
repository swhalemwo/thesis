import csv
import time

from os import listdir
import graph_tool as gt
import subprocess
import os, sys
from datetime import datetime


from graph_tool.all import *
from graph_tool import *  



stage_dir = "/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/stage/"
stage_files = listdir(stage_dir)


for i in stage_files:
    print(i)
    # i = stage_files[20]
    gx = gt.load_graph_from_csv(stage_dir+i, directed=True, string_vals=True, csv_options={'delimiter':'\t'})
    rel_mbids = find_vertex_range(gx, 'in', (100, 10**19))

    # need to set limits
    # should be rather low: want to focus on new ones
    # can't help but think i'm not focusing on what's important here: nobody cares about super small genres
    # -> large forms should get the focus
    # absence of change, stability, locked in place
    # big ones can matter for legitimation?

    # genres are not equals

    mbid_file = stage_dir + "mbids/" + i[0:len(i)-4] + ".csv"

    with open(mbid_file, 'w') as fo:
        wr = csv.writer(fo)
        [wr.writerow([gx.vp.name[i]]) for i in rel_mbids]

## add saving of graph so that i don't have to read them in all the time
    


# python3.6 tags.py album "/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_mbid.csv" "/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_tags1.sqlite"^C

# see how concentration of listening events changes over time


check amount of plays: outdegree
edg_cntr = 0


usrs = find_vertex(gx, 'in', 0)
v_cnt = len(list(gx.vertices()))
# percent of listening events
sum([j.in_degree() for j in rel_mbids])/gx.edge_index_range
# percent of albums
len(rel_mbids)/(v_cnt-len(usrs))



# niche: contribution should be based on in degree

tasks:
- album mbid lists
- mds
- niche size
  need to specify tags
  -> need to know uses of tags per year


