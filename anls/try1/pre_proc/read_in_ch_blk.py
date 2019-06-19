from clickhouse_driver import Client
import csv
import json
import numpy as np
from datetime import datetime
from datetime import timedelta
import random
import os
import argparse
import time
import subprocess
import itertools

def get_log_clean(f):
    """return cleaned log (only songs that exist)"""
    # log_lines = []
    usr_log=[]
    # usr_tstamps=[]
    with open(f, 'r') as fi:
        rdr=csv.reader(fi, delimiter='\t')

        for row in rdr:
            if len(row[3])==36:
                usr_log.append((row[0],row[3]))
    return(usr_log)


def unq_proc_bulk(usr_log, mbid_abbrv_dict):
    """get new additions for mbid_abbrv dict"""
    
    usr_all_songs = [i[2] for i in usr_log]
    usr_unq_songs=list(set(usr_all_songs))

    db_songs_unq = list(mbid_abbrv_dict.keys())

    usr_new_unqs=list(set(usr_unq_songs) - set(db_songs_unq))
    
    cur_cnt=len(db_songs_unq)

    nbrs_add = range(cur_cnt+1, cur_cnt+len(usr_new_unqs)+1)
    unq_ids = ['s'+ str(i) for i in nbrs_add]

    # unq_ids=[]
    # for i in nbrs_add:
    #     print(i)
        # unq_ids.append('s'+ str(i))

    # generate random shit so that partioning is happy

    rnd_parts = random.choices(range(10), k=len(unq_ids))

    add_kv_songs = [(i) for i in zip(usr_new_unqs, unq_ids, rnd_parts)]
    return(add_kv_songs)

def update_mbid_abbrv(mbid_abbrv_dict, add_kv_songs):
    """update song abbrevs dict"""
    for i in add_kv_songs:
        mbid_abbrv_dict[i[0]]=i[1]
    return(mbid_abbrv_dict)


def get_db_songs():
    """create song abbrevs dict"""
    songs_mbid_abbrev = client.execute('select mbid, abbrv from song_info')

    mbid_abbrv_dict={}

    for i in songs_mbid_abbrev:
        # abbrv_mbid_dict[i[1]]=i[0]
        mbid_abbrv_dict[i[0]]=i[1]

    return(mbid_abbrv_dict)


def check_present(usr):
    """check if user is already in db"""
    exsting =client.execute("select Count(*) from logs where usr='" + usr + "'")[0][0]
    if exsting == 0:
        status=False
    else:
        status=True
    return(status)


def bucketer(all_logs, min_dts, max_dts):
    """process time buckets"""
    min_dt = min(min_dts)
    max_dt = max(max_dts)

    delta = max_dt - min_dt

    bucket_dict = {}        
    bkt_cntr = 0
    bkt_size = 95
    
    for i in range(0, delta.days+1, 1):
        bkt_size -=1

        dtx = min_dt + timedelta(days=i)

        bucket_dict[dtx]=bkt_cntr
        
        if bkt_size ==0:
            bkt_cntr+=1
            bkt_size = 95
            
    buckets={}
    for i in range(0, bkt_cntr+1, 1):
        buckets[i] = []

    for i in all_logs:
        bktx = bucket_dict[i[0]]
        buckets[bktx].append(i)

    return(buckets)


client = Client(host='localhost', password='anudora', database='frrl')
# client.execute('drop table tests2')
# client.execute('create table tests2 (tx Date, usr String, song String) engine=MergeTree(tx, tx, 8192)')

if __name__ == '__main__':
    # reads all the logs
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', help='dir with logs')

    args = parser.parse_args()
    log_dir = args.log_dir

    # log_dir = '/media/johannes/D45CF5375CF514C8/Users/johannes/mlhd/0-15/02/'
    # log_dir = '/home/johannes/Downloads/mlhd/06/'
    
    files1=os.listdir(log_dir)
    log_files = [i for i in files1 if i.endswith('.txt')]

    chunk_size = 10

    chunks = [log_files[x:x+chunk_size] for x in range(0, len(log_files), chunk_size)]
    # c=chunks[0]

    print('creating mbid abbrv dict')
    mbid_abbrv_dict=get_db_songs()


    for c in chunks:
        print('new chunk', chunks.index(c)+1, '/', len(chunks))
        # bulk dict construction

        valid_uuids = []
        all_logs = []
        min_dts=[]
        max_dts=[]

        for l in c:
            
            f=log_dir + l
            uuid=l[0:36]

            try:
                ab_uid = client.execute("select abbrv2 from usr_info where uuid='"+uuid + "'")[0][0]

                logx = get_log_clean(f)

                logx2 = [(
                    datetime.date(datetime.utcfromtimestamp(int(i[0]))),
                    ab_uid,
                    i[1]) for i in logx]

                # some logs are reverse: earliest first
                if int(logx[0][0]) > int(logx[-1][0]):

                    min_dts.append(logx2[-1][0])
                    max_dts.append(logx2[0][0])
                    
                    logx2.reverse()

                else:
                    min_dts.append(logx2[0][0])
                    max_dts.append(logx2[-1][0])

                all_logs = all_logs + logx2
                valid_uuids.append(ab_uid)

            except:
                # print('user not in usr_info')
                # need to log to error file
                print('someting wong with', ab_uid)
                continue

        new_addgs = unq_proc_bulk(all_logs, mbid_abbrv_dict)
        
        mbid_abbrv_dict = update_mbid_abbrv(mbid_abbrv_dict, new_addgs)
        print(len(mbid_abbrv_dict))

        # !!!!!! MASTER CAUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        client.execute('insert into song_info values', new_addgs)
        # !!!!!!!MASTER CAUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        all_logs2 = [(i[0], i[1], mbid_abbrv_dict[i[2]]) for i in all_logs]

        # should put it into proper function
        # min/max dt, bkt size has no point to be known
        # bckt dict neither
        # but insertion should always be own function
        # needs rewriting with result of buckets

        buckets = bucketer(all_logs2, min_dts, max_dts)

        for i in buckets.keys():
            # client.execute('insert into tests2 values', buckets[i])
            client.execute('insert into logs values', buckets[i])
