from datetime import datetime
import random
import csv

from clickhouse_driver import Client
client = Client(host='localhost', password='anudora', database='frrl')

# * logs
client.execute('drop table logs')
client.execute('create table logs (time_d Date, usr String, song String) engine=MergeTree(time_d, time_d, 8192)')

# * usr info
client.execute("""
create table usr_info
(uuid String,
age String,
country String,
gender String,
playcount Int32,
age_scrobbles Int32,
user_type String,
registered Int32,
firstscrobble Int32,
lastscrobble Int32)
engine=MergeTree()
PARTITION BY gender
ORDER BY tuple()""")

alter table usr_info drop column abbrv 
alter table usr_info drop column abbrv2

alter table usr_info add column abbrv String
alter table usr_info add column abbrv2 String

alter table usr_info update abbrv=toString(rowNumberInAllBlocks()) where playcount > 0
alter table usr_info update abbrv2=concat(gender, abbrv) where playcount > 0

# abbrv2 has to be based on gender (since usr_info is partitioned by it

# ** samples
# *** sample 1k
uuids = client.execute("""
SELECT uuid, abbrv2 from(
    SELECT distinct(usr) AS abbrv2 FROM logs)
    JOIN usr_info USING abbrv2
    WHERE country = 'US'
"""
)

some_uuids = random.sample(uuids, 1000)

c = 0
uuid_rows = []
for i in some_uuids:
    uuid_row = (i[0], i[1], c % 10)
    uuid_rows.append(uuid_row)
    c +=1

client.execute('DROP TABLE usrs1k')
    
client.execute("""CREATE TABLE usrs1k
(uuid String,
abbrv2 String,
rndm Int8)
engine = MergeTree()
PARTITION BY rndm
ORDER BY tuple()""")

client.execute('INSERT INTO usrs1k VALUES', uuid_rows)




# * playcount testing #


# need to get frequency for all the songs
# similar to counter
# might actually just use counter,

client = Client(host='localhost', password='anudora', database='frrl')
client.execute('show tables')

select song, count(song) as Frequency from logs group by song

client.execute('drop table tests')
client.execute('create table tests (xx String) engine=MergeTree() partition by xx order by tuple()')

client.execute('drop table tests2')
client.execute('create table tests2 (xx String) engine=MergeTree() partition by xx order by tuple()')


testl=[]
for i in range(1000):
    testl.append(random.choice(string.ascii_letters).lower())

client.execute('insert into tests values', testl)
client.execute('insert into tests2 values', list(set(testl)))


client.execute('select xx, count(xx) as Frequency from tests group by xx')

client.execute('select xx, count(xx) as Frequency from tests group by xx join tests2 on tests.xx=tests.xx')

client.execute('drop table tests3')
client.execute('create table tests3 (xx String, yy Int32) engine=MergeTree() partition by xx order by tuple()')

client.execute('insert into tests3 select xx, count(xx) as Frequency from tests group by xx')
client.execute("drop column tests2.xx")


client.execute('create table tests4 (xx String, yy Int32, xx2 String) engine=MergeTree() partition by xx order by tuple()')
client.execute('insert into tests4 select * from tests3 join tests2 on tests3.xx=tests2.xx')

client.execute('alter table tests4 drop column xx2')

client.execute('drop table tests3')




# * actual playcount calculation


client.execute('drop table song_info2')
client.execute('create table song_info2 (song String, cnt Int32, rndm Int32) engine=MergeTree() partition by rndm order by tuple()')

# song_infoer(dd)

# client.execute('select song, count(*), count(*) % 30 from logs group by song limit 3')

# add random shit
client.execute("""insert into song_info2 
select song, count(*), count(*) % 30 from logs group by song""")


# could probably also just select 1 or any number

client.execute('drop table song_info3')

client.execute('create table song_info3 (mbid String, abbrv String, rndm Int32, cnt Int32) engine=MergeTree() partition by rndm order by tuple()')


client.execute("""insert into song_info3 select * from song_info join
(select song as abbrv, cnt from song_info2) using (abbrv)""")



# * tags, tag_sums


# configuration also in ~/Dropbox/gsss/thesis/anls/try1/tag_insert_ch.py

# ** tags
client.execute('drop table tags')
client.execute("""create table tags
(mbid String, tag String, weight Integer)
engine=MergeTree() partition by weight order by tuple()""")

tag_file = "/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split2/1_tags.csv"

batch=[]
c=0
c2 = 0
with open(tag_file, 'r') as fi:
    rdr = csv.reader(fi)
    for row in rdr:
        batch.append([row[0], row[1], int(row[2])])
        c+=1
        c2+=1
        if c == 25000:
            client.execute('insert into tags values', batch)
            c=0
            batch=[]
            print(c2)



# ** tag sums
client.execute("drop table tag_sums")

client.execute("""create table tag_sums (mbid String, tag String, weight Int32, ttl_weight Int32)
engine=MergeTree() partition by weight order by tuple()""")

client.execute("""insert into tag_sums
select * from tags 
join (select mbid, sum(weight) from tags where weight > 10 group by mbid) using (mbid)""")

client.execute("alter table tag_sums add column rel_weight Float32")

client.execute("alter table tag_sums update rel_weight = divide(weight,ttl_weight) where weight > 0")














# * other mb/lfm api related information

# ** addgs
# add lfm id, working mbid (might be same), artist, title, earliest release date,
# - level with highest resolution
# - highest resolution of earliest date
# - if earliest date earlier than higher resolved date: time difference between earliest and highest resolved date (0 if no date or if highest resolved date is earliest)
# - length of release list
# - length of release list with dates

# selection:
# every entry in 1_addgs should be in song_info 3
# but it's not that much data -> can be easily duplicated and retrieved with join queries


client.execute("drop table addgs")

client.execute("""create table addgs (
lfm_id String,
mbid String,
artist String,
title String,

erl_rls Int32,

max_res Int8,
max_res_erl_rls Int8,
t_diff Int32,
len_rls_lst Int16,
len_rls_dt_lst Int16,

rndm Int8)

engine=MergeTree() partition by rndm order by tuple()""")

batch = []

# tr < ttl_addgs.csv -d '\000' > ttl_addgs2.csv
addgs_file = '/home/johannes/mega/gsss/thesis/remotes/ttl/ttl_addgs2.csv'

with open(addgs_file, 'r') as fi:
    rdr = csv.reader(fi)
    c = 0
    for r in rdr:
        try:
            if r[4] == '9999-09-09':
                r[4] = '2099-09-09'

            dt = datetime.date(datetime.strptime(r[4], '%Y-%m-%d'))

            # CH for now only supports dates since start of unix epoch,
            # requiring shenangians (int days since start) to store it

            dt_int = (dt - datetime.date(datetime(1970, 1, 1))).days

            rp = r[0:4] + [dt_int] + [int(i) for i in r[5:]] + random.sample(range(80),1)

            batch.append(rp)
            # client.execute('insert into addgs values',[rp])

            c+=1
            if c % 10000 == 0:
                # break
                client.execute('insert into addgs values', batch)
                batch = []
        except:
            print('misfit')

    client.execute('insert into addgs values', batch)




# ** X_dones_tags
client.execute('drop table dones_tags')

client.execute("""create table dones_tags (
lfm_id String,
orgn String,
len_tag_lst Int8,
lsnrs Int32,
plcnt Int32,
rndm Int8)
engine = MergeTree() partition by rndm order by tuple()""")

# lens = []
# rows = []


# tr < ttl_dones_tags.csv -d '\000' > ttl_dones_tags2.csv
dones_tags_file = '/home/johannes/mega/gsss/thesis/remotes/ttl/ttl_dones_tags2.csv'
# dones_tags_file = '/home/johannes/Dropbox/orgzly/tag_chunks/chunk3/3_dones_tags.csv'

batch = []
c = 0
err_c = 0

with open(dones_tags_file, 'r') as fi:
    rdr = csv.reader(fi)

    for r in rdr:
        if len(r) != 5:
        # if r[1] in ['fail', 'manual', 'lfm', 'mb']:
            err_c+=1

        else:
            # lens.append(int(r[2]))
            rp = r[0:2] + [int(i) for i in r[2:]] + random.sample(range(80),1)
            batch.append(rp)

            c+=1
            if c % 10000 == 0:
                print(c)
                print(err_c)
                # break
                # client.execute('insert into dones_tags values', batch)
                batch =[]
    # except:
    #     print('misfit')
    #     break
            # pass
    client.execute('insert into dones_tags values', batch)

    # there's an entire 200k segment that doesn't have the normal structure -> figure out after break
    # is between 390 and 400k
    # but no idea how 10k rows can increase error counter to 200k... - >break frist
    # is apparently issue of chunk3, maybe it ran with an older version? 

            
# ** acousticbrainz

# there are some ugly duplicates in lfm_ids in addgs, around 176
# no idea where they come from
# point to completely different things
# skip them for now

# maybe something not working with the splits?
# maybe i didn't delete something properly, at the start? 
# or maybe when restarting split scripts with C-c? 

client.execute('drop table acstb')

client.execute("""create table acstb2 (
lfm_id String,
dncblt Float32,
gender Float32,
timb_brt Float32,
tonal Float32,
voice Float32,
mood_acoustic Float32,
mood_aggressive Float32,
mood_electronic Float32,
mood_happy Float32,
mood_party Float32,
mood_relaxed Float32,
mood_sad Float32,
len Float32,
rndm Int8)

engine=MergeTree() partition by rndm order by tuple()""")

avbl_vars = ['id', 'dncblt', 'gender', 'timb_brt', 'tonal', 'voice', 'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy', 'mood_party', 'mood_relaxed', 'mood_sad', 'gnr_dm_alternative', 'gnr_dm_blues', 'gnr_dm_electronic', 'gnr_dm_folkcountry', 'gnr_dm_funksoulrnb', 'gnr_dm_jazz', 'gnr_dm_pop', 'gnr_dm_raphiphop', 'gnr_dm_rock', 'gnr_rm_cla', 'gnr_rm_dan', 'gnr_rm_hip', 'gnr_rm_jaz', 'gnr_rm_pop', 'gnr_rm_rhy', 'gnr_rm_roc', 'gnr_rm_spe', 'gnr_tza_blu', 'gnr_tza_cla', 'gnr_tza_cou', 'gnr_tza_dis', 'gnr_tza_hip', 'gnr_tza_jaz', 'gnr_tza_met', 'gnr_tza_pop', 'gnr_tza_reg', 'gnr_tza_roc', 'mirex_Cluster1', 'mirex_Cluster2', 'mirex_Cluster3', 'mirex_Cluster4', 'mirex_Cluster5', 'length', 'label', 'lang', 'rl_type', 'rls_cri']

chsn_vars = ['dncblt', 'gender', 'timb_brt', 'tonal', 'voice', 'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy', 'mood_party', 'mood_relaxed', 'mood_sad', 'length']

ids = [avbl_vars.index(i) for i in chsn_vars]

acstb_file = '/home/johannes/mega/gsss/thesis/acb/acstbrnz.csv'

'/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/acstbrnz.csv'

import random

c = 0
batch = []
with open(acstb_file, 'r') as fi:
    rdr = csv.reader(fi)
    for r in rdr:

        c+=1
            
        rp = [r[0]] + [round(float(r[k]),5) for k in ids] + random.sample(range(80),1)
        batch.append(rp)

        if c % 10000 == 0:
            client.execute('insert into acstb2 values', batch)
            batch=[]
            print(c)
            
    client.execute('insert into acstb2 values', batch)

    # ids = [r[0] for r in rdr]



