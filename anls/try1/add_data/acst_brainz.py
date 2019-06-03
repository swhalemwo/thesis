# from clickhouse_driver import Client
import csv
import string
import random
from collections import Counter
import json
import urllib.request
import requests
import time
# client = Client(host='localhost', password='anudora', database='frrl')
# top_mbids=client.execute('select mbid from song_info3 order by cnt desc limit 20')





##########################
# retrieving information #
##########################

url = "https://acousticbrainz.org/api/v1/04fa3746-035a-48f1-86b2-3514940c7aaf/high-level"


url="https://acousticbrainz.org/api/v1/high-level?recording_ids=96685213-a25c-4678-9a13-abd9ec81cf35"
url="https://acousticbrainz.org/api/v1/low-level?recording_ids=96685213-a25c-4678-9a13-abd9ec81cf35"
url="https://acousticbrainz.org/api/v1/low-level?recording_ids=2ee68ec3-d85b-4bc9-8f65-3a109af26b5a"

url="https://acousticbrainz.org/api/v1/high-level?recording_ids=c69310f9-e2e5-4fb2-ac26-836913c478d4"




# hm how should i handle mibd insufifficiencies (mlhd id differeing from mbid)
# get mb mbid first?
# i think MB api is kinda fast? also think it has batch requests?
# nope, MB api is slow af (1 request/sec)
# had to rewrite tag retrieval script: split, also store mbid


with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split/4_addgs.csv', 'r') as fi:
    rdr = csv.reader(fi)
    some_mbids = [i[0:2] for i in rdr]


i = some_mbids[9]

uneqs = [i for i in some_mbids if i[0]!=i[1]]

vlds = []
## testing which mbid to use
for i in some_mbids:
# for i in uneqs:           
    # url = 'https://acousticbrainz.org/api/v1/' + i[0] + '/low-level/'
    # url = 'https://acousticbrainz.org/api/v1/' + i + '/low-level/'
    url = 'https://acousticbrainz.org/api/v1/high-level?recording_ids='+ i[1]
       
    with urllib.request.urlopen(url) as url2:
        data = json.loads(url2.read().decode())

    url = 'https://acousticbrainz.org/api/v1/high-level?recording_ids='+ i[0]
    with urllib.request.urlopen(url) as url2:
        data2 = json.loads(url2.read().decode())

    print(len(data), len(data2))

    if len(data)==1:
        vlds.append(i[1])
    # all patterns can exist: most likely that none, than 1 1, than 1 0; 0, 1 very rarely
    # hm not clear how to batch process them
    # if same (85% of cases: i can just take one)
    # is batch processing substantially faster?
    if len(vlds)==60:
        break
    
# batch processing testing
basestr = 'https://acousticbrainz.org/api/v1/high-level?recording_ids='
k = some_mbids[0]

X = 49


teststr = ""
cntr = 0
# for i in some_mbids[0:X]:
for i in vlds[0:X]:    
    teststr = teststr + i

    cntr +=1
    if cntr < X:
        teststr = teststr + ';'
    

url = basestr + teststr
with urllib.request.urlopen(url) as url2:
    data2 = json.loads(url2.read().decode())


def batch_prepper(batch):
    base_str = 'https://acousticbrainz.org/api/v1/high-level?recording_ids='
    cntr = 0
    batch_str = ""
    for i in batch:
        cntr +=1
        batch_str = batch_str + i
        if cntr < len(batch):
            batch_str = batch_str + ';'

    url = base_str + batch_str
    return(url)



# calling with 9: return5, seemingly random order (1,3,4,6,9)
# could just be those that work?
# yup
# seems possible to get input of 48 max
# yup: 49 breaks it even when all valid -> batches of 48

# that poor api omg

# if i[0] != i[1]: send both

# need way to manage mlhd ids

batch = []
mlhd_ids = []

pointers = {}

for i in some_mbids:

    
    if i[0] == i[1]:
        batch.append(i[0])
        mlhd_ids.append(i[0])
    else:
        batch.append(i[0])
        batch.append(i[1])

        mlhd_ids.append(i[0])

        pointers[i[0]] = i[1]
        pointers[i[1]] = i[0]

    if len(batch) > 40:
        batch_str=batch_prepper(batch)
        
        # put into own function later
        with urllib.request.urlopen(batch_str) as url2:
            data2 = json.loads(url2.read().decode())

        for i in mlhd_ids:

            # first easy case: i not in pointers
            if i not in pointers.keys():
            
                if i in data2.keys():
                    print('gotcha')

                else:
                    print('fail')

            # more difficult cases: uneqs -> 4 cases: both, neither, 01, 10
            # i = list(pointers.keys())[2]
            if i in pointers.keys():
                k = pointers[i]

                i_stat = i in data2.keys()
                k_stat = k in data2.keys()

                if i_stat = True and k_stat = True:
                if i_stat = True and k_stat = False:
                if i_stat = False and k_stat = True:
                if i_stat = False and k_stat = False:
                # idk if that's efficient, i'm getting tired
                
                
                
            if i in mlhd

        # break




        process_batch(batch, mlhd_ids, pointers)

    # need to recover 
    





    resp_raw = requests.get(url)
    resp_dict = resp_raw.json()

data[i]['0']['metadata']['audio_properties']['length']



data[i]['0']['highlevel'].keys()
data[i]['0']['highlevel']['timbre']['all']['bright']


6a95c356-2287-417f-a2a7-99bfb3730321



danceability
gender
genre_dortmund
genre_electronic
genre_rosamerica
genre_tzanetakis
ismir04_rhythm
mood_acoustic
mood_aggressive
mood_electronic
mood_happy
mood_party
mood_relaxed
mood_sad
moods_mirex
timbre
tonal_atonal
voice_instrumental

Piazzai:
length,
danceability
main key
scale
frequency of main key
chord progression
scale of chord key
bpm
total count of beats

could make separate tables with song - album - artist
if i want to use some more industry-level explanatory mechanisms

# -> uses high level data, but IS SUBJECT TO RECALCULATION


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


##################
# MB API testing #
##################

# WERKS
# https://musicbrainz.org/ws/2/recording/c69310f9-e2e5-4fb2-ac26-836913c478d4?inc=artists+releases
# # not sure if faster than python api tho
# # NOPE 


# https://musicbrainz.org/ws/2/url/ws/2/recording/4843d67e-e3e3-47d0-813b-d4d9f0cb6d56?inc=artists
# https://musicbrainz.org/ws/2/url/ws/2/recording/0f061025-a50e-44d5-8853-86d9ae3d09b9?inc=artists

# https://musicbrainz.org/ws/2/url/ws/2/recording/c69310f9-e2e5-4fb2-ac26-836913c478d4
# https://musicbrainz.org/ws/2/url/ws/2/recording/96685213-a25c-4678-9a13-abd9ec81cf35



# TIME TESTING

# t1 = time.time()
# for i in range(2000):
#     url = 'https://musicbrainz.org/ws/2/recording/c69310f9-e2e5-4fb2-ac26-836913c478d4?inc=artists+releases&fmt=json'
#     resp_raw = requests.get(url)
#     resp_dict = resp_raw.json()
#     print(len(resp_dict))
# t2 = time.time()

# API starts blocking requests when they're too close to each other

# t1 = time.time()
# for i in range(20):
#     mb_inf = musicbrainzngs.get_recording_by_id('c69310f9-e2e5-4fb2-ac26-836913c478d4', includes=['releases', 'artists'])
#     print(i)
    
# t2 = time.time()

# python api adheres to 1 request/second rule

# mb_inf = musicbrainzngs.get_recording_by_id(tops2, includes=['releases', 'artists'])
