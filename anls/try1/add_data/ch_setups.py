from datetime import datetime
import random

from clickhouse_driver import Client
client = Client(host='localhost', password='anudora', database='frrl')


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

with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split2/1_addgs.csv', 'r') as fi:
    rdr = csv.reader(fi)
    c = 0
    for r in rdr:
        if r[4] == '9999-09-09':
            r[4] = '2099-09-09'
        
        dt = datetime.date(datetime.strptime(r[4], '%Y-%m-%d'))
        dt_int = (dt - datetime.date(datetime(1970, 1, 1))).days

        rp = r[0:4] + [dt_int] + [int(i) for i in r[5:]] + random.sample(range(80),1)

        batch.append(rp)
        # client.execute('insert into addgs values',[rp])

        c+=1
        if c % 5000 == 0:
            client.execute('insert into addgs values', batch)
            batch = []

    client.execute('insert into addgs values', batch)



