import pymonetdb
import time

connection = pymonetdb.connect(username="monetdb", password="monetdb", hostname="localhost", database="tags")
cursor = connection.cursor()

cursor.execute('create table tags (mbid string)')
cursor.execute("alter table tags add weight INTEGER")
cursor.execute("alter table tags add mbid_type string")

connection.commit()

t1=time.time()
dd=cursor.execute('select mbid, weight, mbid_type from tags where weight between 14 and 74')
t2=time.time()

dd2 = cursor.fetchall()
t3=time.time()


# monetdb is actually slow af
# again 19k rows/second

import sqlite3
tag_sqldb="/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_tags1.sqlite"
conn = sqlite3.connect(tag_sqldb)
c=conn.cursor()

t1 = time.time()
# stags = c.execute("select mbid, weight, mbid_type from tags where weight=74 or weight= 59 or weight between 1 and 22").
stags = c.execute("select mbid, weight, mbid_type from tags").fetchall()
t2 = time.time()

with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/dump.csv', 'a') as fo:
    wr = csv.writer(fo)
    wr.writerows(stags)


import monetdblite
conn = monetdblite.connect('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/monet2')
conn = monetdblite.make_connection('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/monet2')

c2 = conn.cursor()

c2.execute('create table tags (mbid string)')
c2.execute("alter table tags add weight INTEGER")
c2.execute("alter table tags add mbid_type string")

c2.execute("insert into tags (mbid, weight, mbid_type) values ('eee', 33, 'iii')")



c2.execute('select * from tags where weight between 14 and 74')
t1 = time.time()
ddx=c2.fetchall()
t2 = time.time()


c2.executemany('INSERT into tags (mbid, weight, mbid_type) values (?, ?, ?)', stags)

c2.execute('INSERT into tags (mbid, weight, mbid_type) values (?, ?, ?)', (stags[300][0], stags[300][1], stags[300][2]))


cntr = 0
cntr2 = 0

for i in stags:
    sql_str = "INSERT into tags (mbid, weight, mbid_type) values ('" + i[0] +"'," + str(i[1]) + ",'" + i[2] + "')"
    c2.execute(sql_str)

    cntr+=1
    cntr2 +=1

    if cntr2 ==1000:
        print(cntr)
        cntr2=0


# monetdblite faster than monetdb 100k vs 20k
# sqlite: seems to depend on query? with range: only 188k/sec
# using multiple combinations with few entries also rather less

# mysql: 167k row/sec with all
# 107k rows/sec with between selection on weight
# sqlite looks best tbh

# import mysql-connector-python
import mysql.connector


import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
    user="root", 
    password="/)3lt@l0",
    database='tagx',
    auth_plugin='mysql_native_password')

cx = mydb.cursor()

cx.execute('create table tags (linkage text primary key)')

cx.execute('show databases')

[print(i) for i in cx]


mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")

cx.execute('create table tagx2 (id INT AUTO_INCREMENT PRIMARY KEY, mbid TEXT, weight INT, mbid_type TEXT)')
cx.execute("insert into tagx2 (mbid, weight, mbid_type) values ('ujuju', 30, 'movie')")
mydb.commit()

sql= "insert into tagx2 (mbid, weight, mbid_type) values (%s, %s, %s)"

cx.executemany(sql, stags)


t1=time.time()
cx.execute("select * from tagx2 where weight between 14 and 74")
d=cx.fetchall()
t2=time.time()


mydb.close()


##############
# CLICKHOUSE #
##############

from clickhouse_driver import Client
client = Client(host='localhost', password='anudora', database='tagz')
client.execute('SHOW DATABASES')

client.execute('use overall')

t1=time.time()
dd = client.execute("select usr, song from tagz.tagx where time_d between '2010-10-01' and '2010-10-01'")
t2 = time.time()


# 444k rows/second with all
# doesn't get worse with selection (tbea between 1 and 10k (448k/sec), between 1 and 5k: 441k/sec
# data also seems to be super compact: 2.1m entries with like 15 columns on just 135 mb
# melanielopez1095@outlook.esm

# actual data: only 118k wtf
# might need to switch to
# question is kinda what's producing bottlenecks, might get faster when it can look up more stuff together?




/home/johannes/Dropbox/gsss/thesis/anls/try1/pre_proc/test/f8561044-28a8-4cbc-b64b-5c4d05aa5e0c.txt

client.execute('use tagz')
client.execute('create table tagz (time date, mbidx String) engine=MergeTree(time, (mbidx, time), 8192)')


client.execute('create table tagz (time date, mbidx String) engine=MergeTree(time, (mbidx, time), 8192)')



date Date MATERIALIZED toDate(Unix_ts)


client.execute('select * from tagz')




how to get unix time stamp to date
- something with materialize
- try to get formating done in awk string

- add as int, convert later
  not clear if it then sorts on it properly
  table would already have data to merge, would then need to update


# https://github.com/yandex/ClickHouse/issues/2802
# ENGINE = MergeTree PARTITION BY toYYYYMMDD(created_date)


