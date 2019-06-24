import csv
import requests
import argparse
import os
import re
import pylast
import time


# mbid='b6ab0f53-9acf-430b-b3d8-8b79b0a28201'



def get_tag_dict(a,t):
    """downloads a dict of tags from lfm"""
    
    str1="http://ws.audioscrobbler.com/2.0/?method="
    # str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"
    str4="&api_key=" + API_KEY + "&format=json"

    str22 = 'track' + ".getTopTags&artist=" + a + '&track=' + t
    qry = str1 + str22 + str4

    resp_raw = requests.get(qry)
    resp_dict = resp_raw.json()

    return(resp_dict)


# mbid = i[1]


def nw_proc_id(mbid):
    sx = network.get_track_by_mbid(mbid)
    return(sx)


def nw_proc_at(a,t):
    sx = network.get_track(a,t)
    return(sx)


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


def get_lfm_a_t(mbid):
    str1="http://ws.audioscrobbler.com/2.0/?method="
    str2='track' + ".getInfo&mbid="    
    # str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"
    str4="&api_key="+ API_KEY + "&format=json"
    
    qry = str1 + str2 + mbid + str4

    resp_raw = requests.get(qry)
    resp_dict = resp_raw.json()

    a = resp_dict['track']['artist']['name']
    t = resp_dict['track']['name']

    return((a,t))

def check_resp(resp_dict, mbid):
    """Checks a song resp dict from lfm API for all kinds of errors:
    - song not found
    - no tags
    - "&" in artist or song name 
    """
    
    if list(resp_dict.keys())[0] == 'error':
        print('oopsie woopsie! ' + str(resp_dict['error']), resp_dict['message'], mbid)

        insert_failed(mbid)
        good=0
        # conn.commit()

    elif list(resp_dict.keys())[0]== 'toptags' and len(resp_dict['toptags']['tag']) ==0:
        print('oopsie woopsie! no tags!', mbid)
        insert_failed(mbid)
        good=0

    else:
        good=1
    return(good)

def at_checker(a, t):
    if re.search('&', a) is not None or re.search('&', t) is not None:
        return('fail')
    else:
        return('pass')


def proc_tags(resp_dict, mbid):
    """Processes resp top tag dict"""

    tags = []

    for i in resp_dict['toptags']['tag']:
        tags.append([mbid, i['name'], i['count']])
            
    return(tags)

# i = arg_list[2]

def get_tag_wrpr(arg_list):
    """wrapper for getting tag dict"""
    
    orgn_dict = {1:'mb_names', 2:'id_lfm_names', 3:"mb_lfm_names"}
    
    cntr = 1
    for i in arg_list:
        try:
            if len(i)==1:
                at = get_lfm_a_t(i[0])
                a = at[0]
                t = at[1]

            else:
                a = i[0]
                t = i[1]
            
            if at_checker(a,t) =='pass':
                tag_dict = get_tag_dict(a, t)
                tag_list = proc_tags(tag_dict, arg_list[1][0])

            else:
                nw_song = nw_proc_at(a,t)
                tag_list = tptg_prcs(nw_song, arg_list[1][0])
                
                # MAYBE SHOULD BE EXPENDED TO look up tags with NW directly if previous 3 approaches fail
                # but requires reworking of flow -> first see how many fails are produced

            return([orgn_dict[cntr], tag_list])

        except:
            cntr +=1
            continue
    else:
        return 'song not found nowhere'

            
def get_tag_dones():

    dones1 = open(DONES_FILE).read()
    dones2 = dones1.split('\n')
    dones3 = [i.split(',') for i in dones2]
    dones = [i[0] for i in dones3]

    # with open(DONES_FILE, 'r') as fi:
    #     rdr = csv.reader(fi)
    #     dones = [row[0] for row in rdr]

    return(dones)

def get_todos(dones):
    """compares todos with dones, might need some optimization"""

    addgs_dict = {}
    
    x1 = open(TODO_FILE).read()
    x2 = x1.split('\n')
    x3 = [i.split(',') for i in x2]

    for row in x3:
        addgs_dict[row[0]] = row[0:4]

    # with open(TODO_FILE, 'r') as fi:
    #     rdr = csv.reader(fi)
    
    #     for row in rdr:
    #         addgs_dict[row[0]] = row[0:4]

    for i in tags_done:
        try:
            addgs_dict.pop(i)
        except:
            pass


    all = [addgs_dict[i] for i in list(addgs_dict.keys()) if len(i) == 36]
    

        # all = [row[0:4] for row in rdr if row[0] not in dones]

        # all = []
        # c = 0
        # for row in rdr:
        #     if row[0] not in dones:
        #         all.append(row[0:4])
                

        #     c+=1
        #     if c % 50 ==0:
        #         print(c)                
        
    return(all)

def save_tags(tags):
    """Saves abbrv-tag-weight list"""
    with open(TAG_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerows(tags)


def insert_dones(dones_obj):
    with open (DONES_FILE, 'a') as fo:
        wr = csv.writer(fo)
        wr.writerow(dones_obj)


def insert_failed_tag(mbid):
    with open(FAILED_FILE, 'a') as fi:
        wr = csv.writer(fi)
        wr.writerow([mbid])


def get_info_at(a,t):
    str1="http://ws.audioscrobbler.com/2.0/?method="
    str2='track.getInfo&artist=' + a
    str3= '&track='+t
    # str4 = '&api_key=607aa0a70e1958439a7de088b66a4561&format=json'
    str4 = '&api_key=' + API_KEY + '&format=json'


    qry = str1 + str2 + str3 + str4

    resp_raw = requests.get(qry)
    resp_dict = resp_raw.json()
    return(resp_dict)

def get_info_id(idx):
    str1="http://ws.audioscrobbler.com/2.0/?method="
    str2='track.getInfo&mbid='
    # str4 = '&api_key=607aa0a70e1958439a7de088b66a4561&format=json'
    str4 = '&api_key=' + API_KEY + '&format=json'

    qry = str1 + str2 + idx + str4
    resp_raw = requests.get(qry)
    resp_dict = resp_raw.json()

    return(resp_dict)

def proc_inf(resp_dict):
    lsnrs = int(resp_dict['track']['listeners'])
    plcnt = int(resp_dict['track']['playcount'])
    return([lsnrs,plcnt])


def get_plcnts(arg_list):
    """gets playcount"""
    cntr = 1
    for i in arg_list:

        try:
            if len(i)==1:
                inf_tag = get_info_id(i[0])
            else:
                inf_tag = get_info_at(i[0], i[1])
                
            return(proc_inf(inf_tag))
        except:
            continue
    else:
        return([0,0])


def get_auth(authfile):
    auths = []
    with open(authfile, 'r') as fi:
        rdr = csv.reader(fi)
        auths = [i for i in rdr]
    return(auths)

    
if __name__ == '__main__':

    # API_KEY = "6ff51b99224a1726d47f686d7fcc8083"
    # API_SECRET="1ba59bdc2b860b8c9f52ac650e3cb6ab"


    parser = argparse.ArgumentParser()
    parser.add_argument('chunk_dir', help = 'working directory, place where all the magic happens')
    parser.add_argument('chunk_nbr', help='chunk number')
    parser.add_argument('authfile', help='lfm keys and secret')
    args = parser.parse_args()

    # !!!!!!!!!!!!!!!!!!!!! THIS IS A DRILL !!!!!!!!!!!!!!!!!!!!!!!!!!
    # chunk_nbr = '10'
    # chunk_dir = '/home/johannes/mega/gsss/thesis/remotes/chunk10/chunk10/'
    # authfile = '/home/johannes/Dropbox/gsss/thesis/anls/try1/authfile.txt'
    # !!!!!!!!!!!!!!!!!!!!! THIS IS A DRILL !!!!!!!!!!!!!!!!!!!!!!!!!!


    end_nigh = 0

    chunk_dir = args.chunk_dir
    chunk_nbr = str(args.chunk_nbr)
    authfile = args.authfile
    
    # chunk_nbr = '5'
    # chunk_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/chunk5/'

    DONES_FILE = chunk_dir + chunk_nbr + '_dones_tags.csv'
    TODO_FILE = chunk_dir + chunk_nbr + '_addgs.csv'
    FAILED_FILE = chunk_dir + chunk_nbr + '_tags_failed.csv'
    TAG_FILE = chunk_dir + chunk_nbr + '_tags.csv'

    open(DONES_FILE, 'a')
    
    tags_done = get_tag_dones()
    todos = get_todos(tags_done)

    cntr = 0

    # authfile = '/home/johannes/Dropbox/gsss/thesis/anls/try1/authfile.txt'
    auths = get_auth(authfile)
    API_KEY = auths[0][0]
    API_SECRET = auths[1][0]
    
    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)




    while len(todos) > 0:
        i = todos[0]

        if i[0] == i[1]:
            arg_list = [[i[2], i[3]], [i[0]]]
        else:
            arg_list = [[i[2], i[3]], [i[0]], [i[1]]]

        plcnt = get_plcnts(arg_list)

        tags = get_tag_wrpr(arg_list)

        if tags == 'song not found nowhere':
            print('do stuff here that processes nonfound songs')
            insert_failed_tag(i[0])
            insert_dones([i[0], 'fail']+plcnt)

        else:
            tag_list = tags[1]
            tag_orgn = tags[0]

            save_tags(tag_list)
            insert_dones([i[0], tag_orgn, len(tag_list)] + plcnt)

        todos.pop(0)
        cntr+=1

        if cntr == 10:
            print(len(todos))
            cntr = 0

        # get new ones
        # might have to fix: when coming to end, it will try to read new ones
        # but there aren't, but still sleep 5 sec each run
        
        if len(todos) < 100 and end_nigh == 0:
            time.sleep(20)
            tags_done = get_tag_dones()
            todos = get_todos(tags_done)

            # if no new ones added (still < 100): stop checking for new ones completely
            if len(todos) < 100:
                end_nigh = 1
                print("THE END IS NIGH")
            # could add another check for 10, but see first
        
# quite much faster, can use mb names really in 99 of cases
# seems to be at 2/sec 
# playcount info call should be in here: probably slows it down to 1/sec, but better here than in 1

# implemented:
# ATM tags22 at 27/20 sec, splt1: 21/20


################################
# pylast vs web API comparison #
################################



# t1=time.time()
# for i in range(20):
# sx = network.get_track_by_mbid(mbid)
# tgs = sx.get_top_tags()
#     print(i)
# t2=time.time()


# t1=time.time()
# for i in range(20):
#     str1="http://ws.audioscrobbler.com/2.0/?method="
#     str2='track' + ".getInfo&mbid="    
#     str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

#     qry = str1 + str2 + mbid + str4

#     resp_raw = requests.get(qry)
#     resp_dict = resp_raw.json()

#     mbid_name = resp_dict['track']['name']
#     artst_name = resp_dict['track']['artist']['name']

#     str22 = 'track' + ".getTopTags&artist=" + artst_name + '&track=' + mbid_name
#     qry = str1 + str22 + str4    
    
#     resp_raw = requests.get(qry)
#     resp_dict = resp_raw.json()
#     print(i)
    
# t2=time.time()

# hm using web api is twice as fast
# question kinda is if it is less accurate
# have to rely on getting top tags with artist name and title
# can't figure out how pylast works

# also playcount and listener count would be nice info to have: sees how well represented my sample is for last fm standards

# aaaaa



# def get_tags(mbid):
    
#     str1="http://ws.audioscrobbler.com/2.0/?method="
#     str2='track' + ".getInfo&mbid="    
#     str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"
    
    
#     qry = str1 + str2 + mbid + str4

#     resp_raw = requests.get(qry)
#     resp_dict = resp_raw.json()

#     if check_resp(resp_dict, mbid) ==1:
#         mbid_name = resp_dict['track']['name']
        
#         artst_name = resp_dict['track']['artist']['name']
        
#         str22 = 'track' + ".getTopTags&artist=" + artst_name + '&track=' + mbid_name
#         qry = str1 + str22 + str4

#         resp_raw = requests.get(qry)
#         resp_dict = resp_raw.json()

#         rtrn = resp_dict 
#     else:
#         rtrn = 'wrong'

#     return(rtrn)

