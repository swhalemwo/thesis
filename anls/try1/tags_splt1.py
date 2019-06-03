import csv
import requests
import argparse
import os
import re
import datetime
from datetime import timedelta
from calendar import monthrange
import musicbrainzngs
musicbrainzngs.set_useragent('mbid_correcter', 0.1, 'johannes.aengenheyster@student.uva.nl')


def date_prcsr(rls_lst):
    """Takes list of releases, returns 
    - earliest date (9999-9-9 if no date)
    - level with highest resolution (0 if no date)
    - highest resolution of earliest date (-1 if no date)
    - if earliest date earlier than higher resolved date: time difference between earliest and highest resolved date (0 if no date or if highest resolved date is earliest)
    - length of release list
    - length of release list with dates
    """
    
    rlss_long=[]
    rlss_medm=[]
    rlss_shrt=[]

    len_rls_lst_ttl = len(rls_lst)
    
    for i in rls_lst:
        try:
            dt = i['date']

            if len(dt) == 10:
                dt2 = [int(i) for i in dt.split('-')]
                dttm = datetime.datetime(dt2[0], dt2[1], dt2[2])
                rlss_long.append(dttm)

            if len(dt) == 7:
                dt2 = [int(i) for i in dt.split('-')]
                max_days = monthrange(dt2[0], dt2[1])[1]
                
                dttm = datetime.datetime(dt2[0], dt2[1], max_days)
                rlss_medm.append(dttm)

            if len(dt) == 4:
                dttm = datetime.datetime(int(dt), 12, 31)
                rlss_shrt.append(dttm)
        except:
            pass

    lowest_lvl = 0
    if len(rlss_shrt) > 0: lowest_lvl=1
    if len(rlss_medm) > 0: lowest_lvl=2
    if len(rlss_long) > 0: lowest_lvl=3

    if lowest_lvl ==0:
        min_ttl = datetime.datetime(9999,9,9)
        max_mbrshp = -1
        len_rls_dts = 0
        tdiff = 0
        # not sure what best default value is: could be e.g. -1, 0, 99
    else:
        all_dts = rlss_long + rlss_medm + rlss_shrt
        min_ttl = min(all_dts)

        len_rls_dts = len(all_dts)

        buckets = [rlss_shrt, rlss_medm, rlss_long]

        min_ttl_mbrshps = []
        cntr = 1
        for i in buckets:
            if min_ttl in i:
                min_ttl_mbrshps.append(cntr)
            cntr +=1

        tdiff = 0 

        max_mbrshp = max(min_ttl_mbrshps)

        if max_mbrshp < lowest_lvl:
            tdiff = min_ttl - min(buckets[lowest_lvl-1])
            tdiff=tdiff.days

            print("DATE ISSUE")
            # maybe add some return thing: return both max(min_ttl_mbrshps) and lowest level
            
    return([min_ttl, lowest_lvl, max_mbrshp, tdiff, len_rls_dts, len_rls_lst_ttl])


def get_mb_inf(idx):
    mb_inf = musicbrainzngs.get_recording_by_id(idx, includes=['releases', 'artists'])
    return(mb_inf)


def get_todos(chunk_nbr, xisting):
    "Get mbids for which to get tags"
    with open(chunk_dir + chunk_nbr) as fi:
        rdr = csv.reader(fi)
        tags_todo = [i[0] for i in rdr]

    tags_todo2= list(set(tags_todo) - set(xisting))
    
    return(tags_todo2)

def get_xisting(chunk_nbr):
    "Gets mbids that have been checked alread (either done or failed)"
    with open(chunk_dir + chunk_nbr + '_dones.csv', 'r') as fi:
        rdr = csv.reader(fi)
        xisting = [i[0] for i in rdr]
    return(xisting)    
        

def save_dones(mbid):
    with open (DONES_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow([mbid])


def save_addgs(prntrow):
    # prntrow = [mbid] + rls_info + [song_orgn]

    with open(ADDGS, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow(prntrow)
    
def insert_failed(mbid):
    "Inserts failed mbids into failed file"
    
    with open(FAILED_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow([mbid])




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_nbr', help='chunk number')
    args = parser.parse_args()

    chunk_nbr = str(args.chunk_nbr)
    # chunk_nbr = '4'

    chunk_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split/'
    FAILED_FILE = chunk_dir + chunk_nbr + '_failed.csv'
    ADDGS = chunk_dir + chunk_nbr + '_addgs.csv'
    DONES_FILE = chunk_dir + chunk_nbr + '_dones.csv'


    open(FAILED_FILE, 'a')
    open(ADDGS, 'a')

    xisting = get_xisting(chunk_nbr)
    todos = get_todos(chunk_nbr, xisting)

    for i in todos:
        try:
            mb_inf = get_mb_inf(i)
        except:
            save_addgs(i, ['mb retrieval failed'], None, 0)
            insert_failed(i)
            continue

        idx = mb_inf['recording']['id']
        a = mb_inf['recording']['artist-credit'][0]['artist']['name']
        t = mb_inf['recording']['title']

        rls_info = date_prcsr(mb_inf['recording']['release-list'])
        rls_info[0] = rls_info[0].strftime('%Y-%m-%d')

        a = 'billy & the news'

        prntrow = [i, idx, a, t] + rls_info
        save_addgs(prntrow)


with open(ADDGS, 'r') as fi:
    rdr =csv.reader(fi)
    res = [row for row in rdr]
        
        

