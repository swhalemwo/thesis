from clickhouse_driver import Client
import csv
import json
import numpy as np
from datetime import datetime
import random
import os
import argparse

# log_dir ="/home/johannes/Dropbox/gsss/thesis/anls/try1/pre_proc/test/"
# client = Client(host='localhost', password='anudora', database='tagz')

# create table usr_info
# (uuid String,
# age String,
# country String,
# gender String,
# playcount Int32,
# age_scrobbles Int32,
# user_type String,
# registered Int32,
# firstscrobble Int32,
# lastscrobble Int32)
# engine=MergeTree()
# PARTITION BY gender
# ORDER BY tuple()

# alter table usr_info update abbrv=toString(rowNumberInAllBlocks()) where playcount > 0
# alter table usr_info update abbrv2=concat('u', abbrv) where playcount > 0

# need table in clickhouse -> for joins
# but one that doesn\'t update so that rownumbers are coherent

# client = Client(host='localhost', password='anudora', database='frrl')
# client.execute('show tables')


# log_file='0455f99c-df41-4650-971e-4c0bda45fb97.txt'
# log_file='c435dfbb-1cb0-4dd0-b323-14edfea09969.txt'

# f=log_dir+log_file

def get_log_clean(f):
    # log_lines = []
    usr_log=[]
    # usr_tstamps=[]
    with open(f, 'r') as fi:
        rdr=csv.reader(fi, delimiter='\t')

        for row in rdr:
            if len(row[3])==36:
                usr_log.append((row[0],row[3]))
    return(usr_log)

# usr_log = get_log_clean(f)


def proc_log_wrt_unqs(usr_log, mbid_abbrv_dict):

    usr_all_songs = [i[1] for i in usr_log]
    usr_unq_songs = np.unique(usr_all_songs)

    # add percentages of valid songs as control to usr_info?
    # but all this should be over time.. could still use it to filter out absolute numbnuts tho
    # number of unique songs can be control
    # is like a shitty measure of omnivorism

    db_songs_unq = list(mbid_abbrv_dict.keys())
    usr_new_unqs=np.setdiff1d(usr_unq_songs, db_songs_unq)

    cur_cnt=len(db_songs_unq)
    # print(cur_cnt)
    
    nbrs_add = range(cur_cnt+1, cur_cnt+len(usr_new_unqs)+1)
    unq_ids = ['s'+ str(i) for i in nbrs_add]

    # generate random shit so that partioning is happy
    rnd_parts = random.choices(range(10), k=len(unq_ids))

    add_kv_songs = [(i) for i in zip(usr_new_unqs, unq_ids, rnd_parts)]

    client.execute('insert into song_info values', add_kv_songs)

    return(add_kv_songs)

# new_addgs = proc_log_wrt_unqs(usr_log, mbid_abbrv_dict)




def update_mbid_abbrv(mbid_abbrv_dict, add_kv_songs):
    for i in add_kv_songs:
        mbid_abbrv_dict[i[0]]=i[1]
    return(mbid_abbrv_dict)
        

# mbid_abbrv_dict = update_mbid_abbrv(mbid_abbrv_dict, new_addgs)

# hope 
# needs to create dict at beginning
# then update it with new unique entries

# first create
# might just be enough to maintain list?
# nope: need to convert songs that are in log but not first -> need entire db songs in mbid_abbrv_dict


def get_db_songs():
    songs_mbid_abbrev = client.execute('select mbid, abbrv from song_info')

    mbid_abbrv_dict={}

    for i in songs_mbid_abbrev:
        # abbrv_mbid_dict[i[1]]=i[0]
        mbid_abbrv_dict[i[0]]=i[1]

    return(mbid_abbrv_dict)

def log_procer(ab_uid, usr_log):
    log_proc = []
    dt_cntr = 0
    dt1 = datetime.utcfromtimestamp(int(usr_log[0][0])).strftime('%Y-%m-%d')

    for i in usr_log:

        dtx = datetime.utcfromtimestamp(int(i[0])).strftime('%Y-%m-%d')
        if dtx != dt1:
            dt_cntr +=1
            dt1 = dtx

        log_proc.append((
            dtx,
            ab_uid,
            mbid_abbrv_dict[i[1]]))

        # if usr_log.index(i) % 100 ==0:
        #     print(dt_cntr)

        if dt_cntr == 95:
            # print('commit')
            retrd_insertion(log_proc)
            log_proc = []
            dt_cntr = 0



def retrd_insertion(log_proc):

    # i think the python interface of clickhouse is broken for dates
    # i think i really have to write to file and then back in WTF

    with open('dumb.csv', 'w') as fo:
        wr = csv.writer(fo)
        wr.writerows(log_proc)

        os.system("clickhouse-client --password anudora --format_csv_delimiter=',' --query='INSERT INTO frrl.logs FORMAT CSV' < dumb.csv")



def check_present(usr):
    exsting =client.execute("select Count(*) from logs where usr='" + usr + "'")[0][0]
    if exsting == 0:
        status=False
    else:
        status=True
    return(status)
        


if __name__ == '__main__':
    # reads all the logs
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', help='type of mbid: one of track, album, artist')

    args = parser.parse_args()
    log_dir = args.log_dir
    
    client = Client(host='localhost', password='anudora', database='frrl')

    files1=os.listdir(log_dir)
    log_files = [i for i in files1 if i.endswith('.txt')]

    mbid_abbrv_dict=get_db_songs()

    for i in log_files:
        print(i)

        f =log_dir + i

        # print(i)

        usr_log = get_log_clean(f)
        new_addgs = proc_log_wrt_unqs(usr_log, mbid_abbrv_dict)
        mbid_abbrv_dict = update_mbid_abbrv(mbid_abbrv_dict, new_addgs)

        uuid=i[0:36]

        try:
            ab_uid = client.execute("select abbrv2 from usr_info where uuid='"+uuid + "'")[0][0]
        except:
            print('user not in usr_info')
            continue


        if check_present(ab_uid)==False:
            log_procer(ab_uid, usr_log)


        if log_files.index(i) % 25 ==0:
            print(log_files.index(i))

uuid='9c70cdf1-c0d6-44ac-a679-a0540f923420'

############
# TESTCASE #
############
# log_dir = '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/00/'


# client.execute('drop table song_info')

# client.execute('create table song_info (mbid String, abbrv String, rndm Int) engine=MergeTree() Partition by rndm order by tuple()')

# log_files =["7b178dae-c3d9-4a2c-8c43-1d9fc051983f.txt","f7cde63c-ef61-45bd-a0e1-58757d9d3a2a.txt",
#             "7b32583e-be8d-454e-88fe-d2e6c526f129.txt","f8316336-39d3-4727-b577-bfe617aa39f4.txt",
#             "7b41c874-2117-4b29-9e38-8870d562dfcf.txt","f90c4e5f-7cfe-46d5-82a9-0fd444981917.txt",
#             "7b755ab4-705f-4c6f-a8cd-c98e56273939.txt","f98bd983-d924-459f-ac59-aa1deddc97d1.txt",
#             "7bbc63f1-4b09-411d-a711-404d805e3900.txt","f9a602c4-c717-48e0-8510-102a9d90318b.txt",
#             "7bc83fd1-1ff2-4e01-b2bb-d27d086d38f6.txt","f9aa6387-3b62-4da6-953e-0f1155869127.txt",
#             "7bce4f82-d2e2-41d3-a4e0-67acffb00807.txt","f9caaf75-b57e-4681-9dd5-5e501f1bbbc7.txt",
#             "7bf7ce03-7ec5-4efa-9bac-d1031faa9f57.txt","fa30e89c-71bc-456f-88f3-3fb2aa149bb8.txt",
#             "7c04fa4f-55ef-48c1-b430-7abdc5909895.txt","fa39d456-bad3-444e-8724-1f1a4eb26c41.txt",
#             "7c33a282-afde-4156-949f-bcd08d17fbbd.txt","fa401f01-f159-4a09-bc3a-bede44975111.txt",
#             "7c369790-9cc2-49c6-af36-b3f31b2c9966.txt","fa5f3ca4-aa8b-4f7d-ac90-bcdf6dfb95b5.txt",
#             "7c9c6e3e-91bf-4ab7-b2a7-ba11be480377.txt","fa7054ba-bc00-43ea-abcb-f04623f850db.txt"]


# mbid_abbrv_dict=get_db_songs()

# for i in log_files:
#     print(i)


#     f =log_dir + i

#     # print(i)

#     usr_log = get_log_clean(f)
#     new_addgs = proc_log_wrt_unqs(usr_log, mbid_abbrv_dict)
#     mbid_abbrv_dict = update_mbid_abbrv(mbid_abbrv_dict, new_addgs)

#     uuid=i[0:36]
#     ab_uid = client.execute("select abbrv2 from usr_info where uuid='"+uuid + "'")[0][0]

#     if check_present(ab_uid)==False:
#         log_procer(ab_uid, usr_log)


# unq(mibd) != unq(abbrs) != count(*)
# abbrvs get fucked up first: 4th fucks it up




    # return(songs_db)
# abbrv_mbid_dict={}
# don't think i have to look up the mbids here












# could do multiple runs: one for reading in all unique songs ->
# wait till it sorts everything out
# assign abbrev once
# then create the timestamp db


# although does it matter?
# can give "default" value when reading in
# then assign song_abbrev everywhere where song_abbrev=default

# could also do it in python if i find a way to bulk insert

# no way to get useful partioning key

# stuff has to be be in RAM completely anyways
# and shouldn't be that much -> maybe like five mill or so




# CREATE TABLE my_fancy_kv_store (s String,  k UInt64)
# ENGINE = Join(ANY, LEFT, s);

# INSERT INTO my_fancy_kv_store VALUES ('abc', 1), ('def', 2);


# SELECT joinGet('my_fancy_kv_store', 'x', 'abc');
# SELECT joinGet('my_fancy_kv_store', 'k', 'def');







# theoretical sense of joins:
# - robustness checks:
# - usr
#   - country
#   - gender ???
#   - playcount > X
#     - playcount of artist with mbid > X
#       i think that should be possible with query
#   - period active
#   - age
#   - and probably a lot of stuff that i can't think of now but that I should be able to implement with it  
# - songs:
#   - awards: are genre effects less important for more prestigous artists? 
#   - artists:
#   -> but overall question is on genre level: not sure if i can meaningfully cut artists


# x = {'a':300, 'b':2000}
# with open(log_dir+'test.json', 'w') as fo:
#     json.dump(x, fo)

# dictionary table format exists
# need to get it working
# might be useful to convert all id tables into dicts?

# idk can they select on row criteria?

# idk i think i'll just use a random partioning key 

# client.execute('drop table logs')
# client.execute('create table logs (time_d Date, usr String, song String) engine=MergeTree(time_d, time_d, 8192)')



# uuid=i[0:36]
# ab_uid = client.execute("select abbrv2 from usr_info where uuid='"+uuid + "'")[0][0]

# if check_present(ab_uid)==False:
#     log_procer(ab_uid, usr_log)

    


