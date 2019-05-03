import csv
import time
from os import listdir
import graph_tool as gt
import subprocess
import os, sys
from datetime import datetime


from graph_tool.all import *
from graph_tool import *  




daet_dir = "/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/"

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


fi = open(daet_dir+'read_in.xxx', 'r', sep='\t', newline='\n')
l1 = fi.readlines()
l2 = [i.split() for i in l1]

g = Graph()
g = g.add_edge_list

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

##################################
# add multiple dirs

rel_dirs = ['00','01', '02', '03', '04', '05']
t1 = 1250000000
t2 = 1260000000

cmnds = ['./anls/try1/pre_proc/args2.sh ' + daet_dir + i + " album " + str(t1) + " " + str(t2) for i in rel_dirs]


[os.system(i) for i in cmnds]


stage_cmnds =['cat ' + daet_dir + i + '/read_in.xxx >> '+ daet_dir + 'stage/stage.xxx' for i in rel_dirs]
[os.system(i) for i in stage_cmnds]


# proc_str = 'cd ' + daet_dir + ' && ./args2.sh'



g = gt.load_graph_from_csv(daet_dir + 'stage/stage.xxx', directed=True, string_vals=True, csv_options={'delimiter':'\t'})
# could even have all write to one file directly, not


# looks fuckng stupid
x = range(1, 35000, 1000)
n=vertex_hist(g, 'out', bins=x)[0].tolist()
n2=vertex_hist(g, 'out', bins=x)[1].tolist()[:len(n)]

plt.bar(n2[1:len(n2)],n[1:len(n)])

plt.bar(n, n2)
plt.show()
plt.close()


# maybe vertex similarity becomes especially unnice for high degree data?
# might be related to calculation: multiple paths get traded against each other ???

with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_mbid.csv', 'a') as fo:
    wr = csv.writer(fo)
    [wr.writerow([g.vp.name[v]]) for v in subs2]


##################### add whole list of dates
print(datetime.utcfromtimestamp(1120000000).strftime('%Y-%m-%d %H:%M:%S'))

stage_dir = daet_dir + "stage/"

d1 = date(2007, 1, 1)


# d1 = date(2013, 1, 1)


for i in range(0, 25):
    d2 = d1 + relativedelta(months=3)
    print(d1, d2)
    # d1 = d2

    d1_ut = int(time.mktime(d1.timetuple()))
    d2_ut = int(time.mktime(d2.timetuple()))

    cmnds = ['./anls/try1/pre_proc/args3.sh ' + daet_dir + i + " album " + str(d1_ut) + " " + str(d2_ut) + " " + daet_dir + "stage/" for i in rel_dirs]    

    [os.system(i) for i in cmnds]

    # os.system(cmnds[0])

 # ./args3.sh /media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/01/ album 1239200000 1250000000 /media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/stage/^C
    
    d1 = d2



from dateutil.relativedelta import relativedelta
x + relativedelta(months=3)


split file at time of reading into bunch of subfiles

awk -F\| '{print>$1}' file1
# could combine afterwards 
