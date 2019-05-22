import sqlite3
import Levenshtein
import numpy as np
import time

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
for i in unq3:
    if unq3.index(i) % 20 ==0 :print(i)
    
    smlrtx = [Levenshtein.distance(k, i) for i in unq3]
    tag_smlrt = np.append(tag_smlrt, [smlrtx], axis=0)

# maybe should sort out tags that have short distance to artists

# use DBSCAN? should also work for connected ones with

# histogram

    

t1 = time.time()

t2 = time.time()



for i in unq2[0:200]:
    print(i)



# test stepwise: only test those in terms of assignments that are similar in terms of structure
# test only those with some frequency 3/5/10 to weed out the weirdest ones
# maybe also some minimum number of weights > 30


