import sqlite3
import Levenshtein
import numpy as np
import time
import matplotlib.pyplot as plt


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

    tag_smlrts.append(smlrtx)
    
    tag_smlrt = np.append(tag_smlrt, [smlrtx], axis=0)

tag_smlrt = np.array(tag_smlrts)


plt.hist(smlrtx, bins=20)
plt.show()



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



    

t1 = time.time()

t2 = time.time()



for i in unq2[0:200]:
    print(i)



# test stepwise: only test those in terms of assignments that are similar in terms of structure
# test only those with some frequency 3/5/10 to weed out the weirdest ones
# maybe also some minimum number of weights > 30


