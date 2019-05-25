import sqlite3
import Levenshtein
import numpy as np
import time
import matplotlib.pyplot as plt
from graph_tool.all import *
from graph_tool import *  



sqlite_file='/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_tags1.sqlite'
conn = sqlite3.connect(sqlite_file)
c=conn.cursor()

unq_tags = c.execute('select tag, count(tag) from tags group by tag').fetchall()

unq_tags = c.execute('select tag from tags x join ( select tag from tags group by tag having count (*) > 30) b on x.tag=b.tag').fetchall()

unq_tags = c.execute('select tag from tags where weight > 10 group by tag having count (*) > 10').fetchall()




unq2 = [x[0].lower() for x in unq_tags]
unq3 = list(set(unq2))


x = unq3[0]

tag_smlrt = np.empty([0, len(unq3)])

tag_smlrts = []
for k in unq3:
    if unq3.index(k) % 20 ==0 :print(k)
    
    # smlrtx = [Levenshtein.distance(k, i) for i in unq3]
    smlrtx = [1-(Levenshtein.jaro_winkler(k, i)) for i in unq3]
    smlrtx = [1-(Levenshtein.jaro(k, i)) for i in unq3]
    smlrtx = [distance.levenshtein(k, i, normalized=True) for i in unq3]

    tag_smlrts.append(smlrtx)
    
    tag_smlrt = np.append(tag_smlrt, [smlrtx], axis=0)

tag_smlrt = np.array(tag_smlrts)


plt.hist(smlrtx, bins=20)
plt.show()

def sim_flter(l, oprtr,thrshld):
    ops = {'<':operator.lt,
           '>':operator.gt,
           '=':operator.eq}

    hits = []
    cntr = 0

    for i in l:
        if ops[oprtr](i, thrshld) ==True:
            hits.append(cntr)
        cntr+=1
    
    return(hits)


hits = sim_flter(smlrtx, '<', 0.5)
[print(unq3[i]) for i in hits]

electro should be same to:
- electronic
- electropop


# maybe i should really just use it in a suuuper way to decrease the computational power needed for actual assignment comparison..
# as in to consider/select/mark those cells of the similarity matrix which are not 999

# still results in the general problem of specific vs general tags: asymmetric proximity -> Marieke

# soft cosine?
# idk i think i would have to split by some delimiter? tags would be documents
# but would see electro-rock and electrorock as something very different?

# electro and elektronisch should be similar
# electro and electro-rap should not be that similar

# maybe combine multiple measurements?

textdistance.ratcliff_obershelp('electro', 'electro-blues')

textdistance.ratcliff_obershelp(string1, string2)
# maybe i really have to do really rely on similar assignments

textdistance.levenshtein.normalized_similarity('arrow','arow')
textdistance.levenshtein.normalized_similarity('electro', 'electro-rap')
textdistance.levenshtein.normalized_similarity('electro', 'electronical')

textdistance.levenshtein.normalized_distance('electro', 'electro-rap')
textdistance.levenshtein.normalized_similarity('electro', 'electronical')

textdistance.levenshtein.normalized_distance('ro', 'ro-rap')
textdistance.levenshtein.normalized_similarity('electro', 'electronical')

t1 = time.time()
x = [distance.levenshtein('electrro', 'electro-rap', normalized=True) for i in range(5000)]
t2=time.time()

sims = list(filter(lambda x: x < 0.3, smlrtx))

from sklearn.cluster import DBSCAN
clstr = DBSCAN(eps = 2, min_samples=2, metric='precomputed', leaf_size = 1).fit(tag_smlrt)
# , leaf_size=10)

len(Counter(clstr.labels_))
Counter(clstr.labels_)

def clstr_lablr(mbrshps):

    cntr = 0
    label_dict = {}

    for i in mbrshps:
        if i in label_dict.keys():
            label_dict[i].append(cntr)
        else:
            label_dict[i] = [cntr]
        cntr +=1

    for i in list(label_dict.keys()):
        if len(label_dict[i])==1:
            label_dict.pop(i)

    return(label_dict)



# idk if DBSCAN is so good: don't want long connected clusters where distant but connected elements nothing in common
# hclust
 




# hm not sure what to use for eps
# 1 might not catch electro stuff




from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=5, affinity='precomputed', linkage='complete', compute_full_tree=True)  
AHC = cluster.fit_predict(tag_smlrt)


label_dict = clstr_lablr(AHC)

[print(unq3[i]) for i in label_dict[524]]
len(Counter(AHC))

# should incorporate letters that are the same: electro is more similar to electronic than es to xo


# maybe should sort out tags that have short distance to artists

# use DBSCAN? should also work for connected ones with

# think i should not be too restrictive first: similarity with weights should still split of quite something:
# can't put things together, only pull apart
# OTOH limiting it as much as possible with pure semantic probably reduces the mbid similarity by orders of magnitude



    





# test stepwise: only test those in terms of assignments that are similar in terms of structure
# test only those with some frequency 3/5/10 to weed out the weirdest ones
# maybe also some minimum number of weights > 30


####################
# tag mbid network #
####################

raw = c.execute('select tag, weight, weit_pct from tags2').fetchall()
raw_weits = [i[1] for i in raw]
raw_pcts = [i[2] for i in raw]


# plt.hist(raw_weits, bins=50)
plt.hist(raw_pcts, bins=50)
plt.show()


# peaks at 50, 33, 25 -> all those don't have many
# 44.3k uniques, but +80k values of 100 -> soooo many albums that have so few tags that multiple values at 100, or those that are only met with typical divisons
c.execute('select count(distinct(mbid)) from tags2').fetchall()


raw2 = c.execute('select mbid, tag, weit_pct from tags2 where weit_pct > 0.05 order by weit_pct desc').fetchall()
raw2 = c.execute('select mbid, tag, weit_pct from tags2 where weit_pct > 0.05 order by mbid').fetchall()

# xx = c.execute("select mbid, tag, weit_pct from tags2 where weit_pct > 0.05 and mbid='00007f96-14a8-43e8-955d-0b00323a53bd'").fetchall()

proc = []

t1=time.time()
for i in raw2:
    link = []
    # print(round(i[2]*10))
    # print(round(i[2]*20))
    for k in range(round(i[2]*20)):
        # link.append([i[0], i[1]])
        proc.append([i[0], i[1]])

    # proc = proc + link
    # proc.append(link)

t2=time.time()

gx = Graph()
idx = gx.add_edge_list(proc, string_vals=True, hashed=True)

vx1 = find_vertex(gx, idx, 'electro-pop')[0]
vx2 = find_vertex(gx, idx, 'electropop')[0]

vx1 = find_vertex(gx, idx, 'urban')[0]
vx2 = find_vertex(gx, idx, 'hip hop')[0]


vx1.in_degree()
vx2.in_degree()


itrb = [(int(vx1), int(vx2)), (int(vx2), int(vx1))]

# 21179

# [print(i) for i in gx.vertex(63711).out_neighbors()]

# itrb = [(214, 1198)]
# itrb2 = [(1198, 214)]


# need to get more with electro tags 

slart = vertex_similarity(GraphView(gx, reversed=True), 'jaccard', vertex_pairs=itrb)
min(slart)/max(slart)
max(slart)/min(slart)

# idk i'm kinda stuck



len(set(vx1.in_neighbors()) & set(vx2.in_neighbors()))

# also needs to incorporate actual playcount, otherwise some super small shit mbids get weighted too much
# no idk how to tho




###################
# rescale weights #
###################

c.execute('select count(distinct(mbid)) from tags').fetchall()

c.execute('drop table ttl_weits')
c.execute('create table ttl_weits (mbid text primary key)')
c.execute("ALTER TABLE ttl_weits ADD COLUMN 'weight_ttl' INTEGER")
c.execute("ALTER TABLE ttl_weits ADD COLUMN 'mean_weit' FLOAT")
c.execute("ALTER TABLE ttl_weits ADD COLUMN 'unq_tags' FLOAT")
c.execute("ALTER TABLE ttl_weits ADD COLUMN 'unq_weits' FLOAT")

conn.commit()

# c.execute('insert into ttl_weits select mbid, sum (weight) from tags group by mbid')
# c.execute('insert into ttl_weits select mbid, sum (weight) from tags where weight > 15 group by mbid')
# c.execute('insert into ttl_weits select mbid, sum (weight), avg(weight) from tags where weight > 15 group by mbid')

c.execute('insert into ttl_weits select mbid, sum (weight), avg(weight),count(distinct(tag)), count(distinct(weight)) from tags where weight > 15 group by mbid')


# weits = c.execute('select weight_ttl from ttl_weits').fetchall()
# c.execute('select weight_ttl from ttl_weits where mean_weit < 100').fetchall()

weits = c.execute('select weight_ttl from ttl_weits where mean_weit < 100 and unq_tags > 3').fetchall()
weits2 = [i[0] for i in weits if i[0] < 1500]

plt.hist(weits2, bins=50)
plt.show()

c.execute("select count(*) from ttl_weits where mean_weit < 100 and unq_tags > 3").fetchall()
# 10k have all weights with 100
# 2k tags (6%) additionally have less than 4 unique tags
# 18k additionallly have less than 4 uniqe weights -> great way to filter songs

# may need to select songs based on more criteria:
# - like at least 2/5/8 distinct tags
# - not all weights 100: implies that mean is 100
# not 


# c.execute("select mbid, sum (weight) from tags where weight > 30 and mbid = '0002f642-03a1-4866-bc80-cef3784fd143'").fetchall()


# c.execute("select * from tags where mbid = '0002f642-03a1-4866-bc80-cef3784fd143'").fetchall()
# c.execute('select count(*) from tags where weight > 15').fetchall()

# conn.commit()


# c.execute("select * from ttl_weits where mbid='0002f642-03a1-4866-bc80-cef3784fd143'").fetchall()
# c.execute("select * from ttl_weits where mbid='0011fada-6177-42c0-a3a4-880cd2a153be'").fetchall()

# not getting ttl weight column
# dd = c.execute('select tags.link, tags.mbid, tags.tag, tags.weight from tags cross join ttl_weits on ttl_weits.mbid=tags.mbid limit 3').fetchall()


# columns to select have to be in first select statement already 
dd = c.execute("""select tags.link, tags.mbid, tags.tag, tags.weight, ttl_weits.weight_ttl from tags
        join ttl_weits using (mbid) limit 3""").fetchall()

dd = c.execute("""select link, mbid, tag, weight, weight_ttl from tags
        join (select mbid, weight_ttl from ttl_weits) 
        using (mbid) limit 3""").fetchall()


# does that mean sql statement is executed backwards? idfk

c.execute('drop table tags2')
c.execute("""CREATE TABLE tags2 (link TEXT PRIMARY KEY, 
                                'mbid' TEXT, 
                                'tag' TEXT, 
                                'weight' INTEGER,  
                                weit_ttl Integer)""")
conn.commit()



# c.execute('insert into tags2 select tags.link, tags.mbid, tags.tag, tags.weight, ttl_weits.weight_ttl from tags join ttl_weits using (mbid)')

c.execute("""insert into tags2 
select link, mbid, tag, weight, weight_ttl from tags
join 
(select mbid, weight_ttl from ttl_weits where mean_weit < 100 and unq_tags > 3 and unq_weits > 2)
using(mbid) 
where weight > 15""")
conn.commit()

c.execute("""insert into tags2 select link, mbid, tag, weight, weight_ttl from tags
join (select mbid, weight_ttl from ttl_weits)
using(mbid) where weight > 15""")
conn.commit()

# hm not clear if i should filter songs
# that's what i'm doing when filtering in the join statement
# still no solution for incorporating viewer count
# -> should rather weigh by playcount than filter on album characteristics

# could add it as another edge characteristic, like an additional edge per X plays
# idk
# it conflates tag partiality and wealth/resources
# 

c.execute('select count(distinct(mbid)) from tags2').fetchall()
c.execute('select count(*) from tags2').fetchall()



c.execute('alter table tags2 add column weit_pct Float')
conn.commit()



c.execute('select cast(weight as float)/cast(weit_ttl as float) as res from tags2 limit 3').fetchall()

c.execute('update tags2 set weit_pct = cast(weight as float)/cast(weit_ttl as float)')

# i think 10 is good cutoff


# Select total_percent / no_of_scren as 'result' From yourTableName




# dd = c.execute('select * from ttl_weits limit 3').fetchall()

# CREATE TABLE tags (link TEXT PRIMARY KEY, 'mbid' TEXT, 'tag' TEXT, 'weight' INTEGER, 'mbid_type' TEXT, weit_ttl Integer)

mbid_weits = c.execute('select sum (weight), mbid from tags group by mbid').fetchall()

# mbid_weits = c.execute('select mbid, sum (weight) from tags group by mbid as xxx').fetchall()


# group by tag having count (*) > 10').fetchall()

# maybe with update table? 
c.execute('alter table tags add column weit_ttl Integer')
conn.commit()


# c.executemany('update tags set weit_ttl=? WHERE mbid = ?', mbid_weits)

cntr = 1
for i in mbid_weits:

    c.execute('update tags set weit_ttl=? WHERE mbid = ?', i)
    if cntr % 10 ==0:
        print(i)
        conn.commit()
    cntr +=1

dones = c.execute('select count(distinct(mbid)) from tags where weit_ttl > 0').fetchall()

8f396007-1903-3ed3-888c-4eee72e3bc7c-soul

INSERT INTO new_table
SELECT * FROM table1 CROSS JOIN table2;

# make separate table, join that wit


