import csv
import time
from os import listdir
import graph_tool as gt
import subprocess
import os, sys

from graph_tool.all import *
from graph_tool import *  




daet_dir = "/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/01/"

files=listdir(daet_dir)

g = gt.Graph()
lsn_dt=g.new_edge_property("int")


t1 = time.time()
for i in files:
    # meth 1: awk
    # meth 2: read in everything, subset

    usr_elist = []
    usr_elist2 = []
    # lsn_dts = []
    i_name = i[0:36]
    with open(daet_dir + i, 'r') as fi:
        rdr = csv.reader(fi, delimiter = "\t")

        for row in rdr:
            if int(row[0]) > 1239200000 and int(row[0]) < 1259400000:
                usr_elist.append((i_name, row[2]))
                # usr_elist2.append((i_name, row[2], int(row[0])))
                # lsn_dts.append(row[0])

    print(len(usr_elist))
    some_map = g.add_edge_list(usr_elist, string_vals=True, hashed=True)
t2 = time.time()

# eprops = [lsn_dt]

# don't need time info in graph, selection is before



# use awk first
# awk '$1 >= 1239200000 && $1 <= 1259400000' *.txt >> read_in.txt

# requires filename as output
# awk '$1 > 5 && $1 < 20' *.txt

t3 = time.time()

proc_str = 'cd ' + daet_dir + ' && ./sorter.sh'

os.system(proc_str)

g2 = gt.load_graph_from_csv('/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/01/read_in.xxx', directed=True, string_vals=True, csv_options={'delimiter':'\t'})

t4=time.time()


subs=find_vertex_range(g2, 'in', (500, 1000000))
g2.vp.name[subs]

[print(g2.vp.name[i]) for i in subs]
