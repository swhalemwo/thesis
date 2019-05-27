import csv
import requests
import argparse
# from clickhouse_driver import Client
import os
import re

def get_tags(mbid):
    
    # mbid = '94a2a5ba-390d-4dd9-9f69-3e314f8b9d98'
    # mbid= '0e37764b-0726-4a15-8c16-abc07c4ea033'

    # mbid = i
    mbid = failed[300]

    mbid2 = musicbrainzngs.get_recording_by_id(failed[300])['recording']['id']
    mbid = mbid2

    

    musicbrainzngs.get_recording_by_id(mbid2)['recording']['id']


    str1="http://ws.audioscrobbler.com/2.0/?method="
    str2='track' + ".getInfo&mbid="    
    str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"
    
    
    
    qry = str1 + str2 + mbid + str4

    resp_raw = requests.get(qry)
    resp_dict = resp_raw.json()

    if check_resp(resp_dict, mbid) ==1:
        mbid_name = resp_dict['track']['name']
        
        artst_name = resp_dict['track']['artist']['name']
        
        # str22 = 'track' + ".getTopTags&artist=" + "'" + artst_name + "'" + '&track=' + mbid_name
        str22 = 'track' + ".getTopTags&artist=" + artst_name + '&track=' + mbid_name

        qry = str1 + str22 + str4

        resp_raw = requests.get(qry)
        resp_dict = resp_raw.json()

        rtrn = resp_dict 
    else:
        rtrn = 'wrong'

    return(rtrn)

def proc_tags(resp_dict, mbid):
    # mbid_name = resp_dict['toptags']['@attr'][mbid_type]
    # name = resp_dict['toptags']['@attr']['artist']

    tags = []

    for i in resp_dict['toptags']['tag']:
        tags.append([mbid, i['name'], i['count']])
            
            # do sorting later, better save everything here
            # all those useless tags might matter too

    with open(TAG_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(tags)

        # artist name in quotation marks as otherwise commas would break csv
        # tags can't have commas afaik


def check_resp(resp_dict, mbid):
        if list(resp_dict.keys())[0] == 'error':
            print('oopsie woopsie! ' + str(resp_dict['error']), resp_dict['message'], mbid)

            # c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
            insert_failed(mbid)
            good=0
            # conn.commit()

        elif list(resp_dict.keys())[0]== 'toptags' and len(resp_dict['toptags']['tag']) ==0:
            print('oopsie woopsie! no tags!', mbid)
            insert_failed(mbid)
            # c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
            good=0

        elif re.search('&', resp_dict['track']['artist']['name']) is not None:
            good = 0

        elif re.search('&', resp_dict['track']['name']) is not None:
            good = 0
            
        else:
            good=1
        return(good)

def insert_failed(mbid):
    with open(FAILED_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow([mbid])


def get_todos(chunk_nbr, xisting):

    with open(chunk_dir + chunk_nbr) as fi:
        rdr = csv.reader(fi)
        tags_todo = [i[0] for i in rdr]

    tags_todo2= list(set(tags_todo) - set(xisting))
    
    return(tags_todo2)

def get_xisting(chunk_nbr):
    with open(chunk_dir + chunk_nbr + '_failed.csv', 'r') as fi:
        rdr = csv.reader(fi)
        failed = [i[0] for i in rdr]

    with open(chunk_dir + chunk_nbr + '_tags.csv', 'r') as fi:
        rdr = csv.reader(fi)
        dones_raw = [i[0] for i in rdr]
        dones = list(set(dones_raw))

        xisting = dones + failed
        return(xisting)

# SETUP on runtime


# chunk_nbr = '1'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_nbr', help='chunk number')
    args = parser.parse_args()

    chunk_nbr = str(args.chunk_nbr)
    
    chunk_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/'
    FAILED_FILE = chunk_dir + chunk_nbr + '_failed.csv'
    TAG_FILE=chunk_dir + chunk_nbr + '_tags.csv'
    
    open(FAILED_FILE, 'a')
    open(TAG_FILE, 'a')

    xisting = get_xisting(chunk_nbr)

    todos = get_todos(chunk_nbr, xisting)

    for i in todos:
        resp_dict=get_tags(i)
        if type(resp_dict) == type({}):
            proc_tags(resp_dict, i)

        if todos.index(i) % 10 ==0:
            print(todos.index(i), len(todos), todos.index(i)/len(todos))

# too much shit is missing!
# use musicbrainz to get artist and title 
# might be that many of my current song info's refer to same song
# probably have to clean up logs..
# can make dict with original as key and mb mbid as value
# dicts are quite cheap so might not be an issue
# but means I need to 


###############
# basic setup #
###############

## clickhouseify

# client = Client(host='localhost', password='anudora', database='frrl')
# client.execute('show tables')


# client.execute('create table songs_tags (mbid String, tag String, weight_abs Int32, weight_pct Float32) engine=MergeTree() partition by weight_abs order by tuple()')

# client.execute('create table tags_failed (mbid String, rndm Int32) engine=MergeTree() partition by rndm order by tuple()')

# maybe good idea to split up songs?
# yup: can then be easily distributed

# 250k chunks?

# mbids_ordrd = client.execute('select mbid from song_info3 order by cnt desc')

# cntr = 0
# chunk_cntr = 1
# chunk= []
# for i in mbids_ordrd:
#     chunk.append(i)

#     if cntr == 200000:
#         with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/' + str(chunk_cntr), 'w') as fo:
#             wr = csv.writer(fo)
#             wr.writerows(chunk)

#             chunk = []
#             chunk_cntr +=1
#             cntr = 0
#     cntr+=1


# chunk_cntr +=1
# with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/' + str(chunk_cntr), 'w') as fo:
#             wr = csv.writer(fo)
#             wr.writerows(chunk)
