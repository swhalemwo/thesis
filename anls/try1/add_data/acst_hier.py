import csv
import json
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import math


from scipy.stats import entropy
from sklearn import preprocessing


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

df_tags = pd.DataFrame(rows_tags, columns=['mbid', 'tag', 'rel_weight', 'cnt'])

songs_tags = df_tags['mbid']
len(np.unique(songs_tags))

# select songs that have to be deleted
songs_tbp = list(set(songs_acst) - set(songs_tags))

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


    




# tags = tag_df['tag']
# len(np.unique(tags))

# doesn't seem to make such a difference to include songs cnt > 400: 484k vs 453k 
# presumably because most songs that fulfill the tag requirements (also fulfil the song requirements?)


df = pd.DataFrame(row_list)
# jfc so much faster
# ask marieke
    

    

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
