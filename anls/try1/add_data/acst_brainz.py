from clickhouse_driver import Client
import string
import random
from collections import Counter



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


##########
# actual #
##########


def song_infoer(dd):
    part_cntr = 0
    part = []
    part1 = dd[0]

    for i in dd:
        if i[1] != part1:
            part_cntr +=1
            part1 = i[1]

        part.append(i)

        if part_cntr == 95 or len(part) > 50000:
            client.execute('insert into song_info2 values', part)
            print('commit')
            print(dd.index(i))
            print(part_cntr)
                
            part=[]
            part_cntr=0

    if len(part) > 0:
        client.execute('insert into song_info2 values', part)


client.execute('drop table song_info2')
client.execute('create table song_info2 (song String, cnt Int32) engine=MergeTree() partition by cnt order by tuple()')


song_infoer(dd)



client.execute('drop table song_info3')
client.execute('create table song_info3 (mbid String, abbrv String, rndm Int32, song String, cnt Int32) engine=MergeTree() partition by rndm order by tuple()')

client.execute('insert into song_info3 select * from song_info join song_info2 on song_info.abbrv=song_info2.song')
