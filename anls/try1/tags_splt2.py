import csv
import requests
import argparse
import os
import re
import datetime
from calendar import monthrange
import pylast

def get_song_obj(arg_list, default = None):
    """More sophisticated calling of different ways to get song object"""
    orgn_dict = {1:'lfm', 2:'mb', 3:"manual"}
    fun_list = [network.get_track_by_mbid, network.get_track_by_mbid, network.get_track]
  
    cntr = 1
    for f in fun_list:

        argx = arg_list[cntr-1]
        try:
            sx = f(*argx)

            return([orgn_dict[cntr], sx])
        except:
            cntr +=1
            continue
    else:
        return 'song not found nowhere'

def tptg_prcs(song, mbid):
        """filters tags and weights out of song object"""
    try:
        song_tags = song.get_top_tags()
        tag_list = []
        for k in song_tags:
            tag_list.append([mbid, k.item.name, int(k.weight)])
        return(tag_list)

    except:
        print(mbid, "SUPER WRONG")
        return([])
    
def save_tags(tags):
    """Saves abbrv-tag-weight list"""
    with open(TAG_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(tags)


def insert_dones(mbid):
    with open (DONES_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow([mbid])


def get_todos():
    with open(TODO_FILE, 'r') as fi:
        rdr = csv.reader(fi)
        all = [row[0:4] for row in rdr if row[0] not in dones]

def get_tag_dones():
    with open(DONES_FILE, 'r') as fi:
        rdr = csv.reader(fi)
        dones = [row[0] for row in rdr]

    return(dones)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_nbr', help='chunk number')
    args = parser.parse_args()

    chunk_nbr = str(args.chunk_nbr)
    # chunk_nbr = '4'

    chunk_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split/'
    DONES_FILE = chunk_dir + 'dones_tags.csv'
    TODO_FILE = chunk_dir + chunk_nbr + '_addgs.csv'

    # has to continuously read new stuff ->
    # should continuosly add to some list
    # remove from list when processed, save in 4_tags_done
    # run as long as list has entries


testl = [3,1,2, 5, 2,6]

while len(testl) > 1:
    print(len(testl))

    testl.pop(0)
    time.sleep(0.3)
    
    if len(testl) == 2:
        testl = testl + random.sample(range(0,20),random.sample(range(2,20),1)[0])









    
    

