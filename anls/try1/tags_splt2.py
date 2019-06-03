import csv
import requests
import argparse
import os
import re
import datetime
from calendar import monthrange
import pylast
API_KEY = "6ff51b99224a1726d47f686d7fcc8083"
API_SECRET="1ba59bdc2b860b8c9f52ac650e3cb6ab"
network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)


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


def insert_dones(dones_obj):
    with open (DONES_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow(dones_obj)


def get_todos(dones):
    with open(TODO_FILE, 'r') as fi:
        rdr = csv.reader(fi)
        all = [row[0:4] for row in rdr if row[0] not in dones]
    return(all)

def get_tag_dones():
    with open(DONES_FILE, 'r') as fi:
        rdr = csv.reader(fi)
        dones = [row[0] for row in rdr]

    return(dones)

def insert_failed_tag(mbid):
    with open(FAILED_FILE, 'r') as fi:
        wr = csv.writer(fi)
        wr.writerow([mbid])

def updtr(todos, i, intrvl):
    dings = todos.index(i)
    if dings % intrvl == 0:
        print(dings, len(todos), dings/len(todos))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_nbr', help='chunk number')
    args = parser.parse_args()

    chunk_nbr = str(args.chunk_nbr)
    # chunk_nbr = '4'

    chunk_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split/'
    DONES_FILE = chunk_dir + chunk_nbr + '_dones_tags.csv'
    TODO_FILE = chunk_dir + chunk_nbr + '_addgs.csv'
    FAILED_FILE = chunk_dir + chunk_nbr + '_tags_failed.csv'
    TAG_FILE = chunk_dir + chunk_nbr + '_tags.csv'

    # has to continuously read new stuff ->
    # should continuosly add to some list
    # remove from list when processed, save in 4_tags_done
    # run as long as list has entries

    open(DONES_FILE, 'a')

    tags_done = get_tag_dones()
    todos = get_todos(tags_done)

    # todos = todos[0:120]

    cntr = 0

    while len(todos) > 0:
        i = todos[0]
        arg_list = [[i[0]], [i[1]], [i[2], i[3]]]
        song_output = get_song_obj(arg_list)

        if song_output == 'song not found nowhere':
            print('do stuff here that processes nonfound songs')
            insert_failed_tag(i[0])
            insert_dones([i[0], 'fail'])

        else:
            song = song_output[1]
            song_orgn = song_output[0]

            song_tags = tptg_prcs(song, i[0])

            save_tags(song_tags)
            insert_dones([i[0], song_orgn, len(song_tags)])
            
        
        
        todos.pop(0)
        cntr+=1

        if cntr == 10:
            print(len(todos))
            cntr = 0

        # get new ones
        if len(todos) < 100:
            tags_done = get_tag_dones()
            todos = get_todos(tags_done)
            



            # hm is origin worth saving?
            # could put it tags_done

            


###################################
# testing of while loops/list pop #
###################################

# testl = [3,1,2, 5, 2,6]

# while len(testl) > 1:
#     print(len(testl))

#     testl.pop(0)
#     time.sleep(0.3)
    
#     if len(testl) == 2:
#         testl = testl + random.sample(range(0,20),random.sample(range(2,20),1)[0])









    
    

