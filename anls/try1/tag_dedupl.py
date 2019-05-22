import sqlite3

sqlite_file='/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_tags1.sqlite'
conn = sqlite3.connect(sqlite_file)
c=conn.cursor()

unq_tags = c.execute('select distinct(tag) from tags').fetchall()

unq2 = [x[0].lower() for x in unq_tags]
unq3 = list(set(unq2))

for i in unq3[0:100]:
    print(i)

# test stepwise: only test those in terms of assignments that are similar in terms of structure
# 

