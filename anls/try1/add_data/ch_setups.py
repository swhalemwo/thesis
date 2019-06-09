#####################
# playcount testing #
#####################

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



################################
# actual playcount calculation #
################################

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
