import csv
import requests
import argparse
import sqlite3

def check_existing(c):
    q=[i[0] for i in c.execute('SELECT DISTINCT mbid from tags').fetchall()]
    f=[i[0] for i in c.execute('SELECT mbid from failed').fetchall()]

    [q.append(i) for i in f]
    
    return(q)

    # with open(tagfile, 'r') as fi:
    #     rdr = csv.reader(fi)
    #     exstng_mbids = [row[0] for row in rdr]

        
    # return(exstng_mbids)


def load_todo(infile): 
    with open(infile, 'r') as fi:
        rdr = csv.reader(fi)
        mbids_todo = [row[0] for row in rdr]
    return(mbids_todo)
 

# could also add all the todo ones as an table
# don't really see the advantage tho: not really operated on besides getting all
# also means i'll have to clean up
# think it's easier to check for duplicates elsehow


def check_resp(resp_dict, mbid):
        if list(resp_dict.keys())[0] == 'error':
            print('oopsie woopsie! ' + str(resp_dict['error']), resp_dict['message'])
            c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
            good=0
            # conn.commit()

        elif list(resp_dict.keys())[0]== 'toptags' and len(resp_dict['toptags']['tag']) ==0:
            print('oopsie woopsie! no tags!')
            c.execute('INSERT OR IGNORE INTO failed (mbid) VALUES (?)', (mbid,))
            good=0

        else:
            good=1

        return(good)


def get_tags(mbid, mbid_type, c):

    # rage against the machine: artist, album, song
    # mbid = '3798b104-01cb-484c-a3b0-56adc6399b80'
    # mbid = '10cf6ada-ef3f-3cd3-9a7b-bdc15db2dddf'
    # mbid = 'fa2e0148-835d-473a-babf-c2a3188879d2'

    # mbid_type = 'artist'
    # mbid_type = 'album'
    # mbid_type = 'track'

    if mbid_type == 'artist':
        str1="http://ws.audioscrobbler.com/2.0/?method="
        str2=mbid_type +".getTopTags&mbid="
        str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

        # artst = '936db8ea-2d4b-4d2b-9665-ad8fdecac835'
        qry = str1 + str2 + mbid + str4

        resp_raw = requests.get(qry)
        resp_dict = resp_raw.json()


    else:
        # get info first
        str1="http://ws.audioscrobbler.com/2.0/?method="
        str2=mbid_type + ".getInfo&mbid="    
        str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

        qry = str1 + str2 + mbid + str4
        resp_raw = requests.get(qry)
        resp_dict = resp_raw.json()

        # if list(resp_dict.keys())[0] == 'error':
        #     print(resp_dict['error'], resp_dict['message'])
        #     fail = 1

        if check_resp(resp_dict, mbid) ==1:
            mbid_name = resp_dict[mbid_type]['name']

            if mbid_type=='album':
                artst_name = resp_dict['album']['artist']                

            if mbid_type=='track':
                artst_name = resp_dict['track']['artist']['name']

            str22 = mbid_type + ".getTopTags&artist=" + artst_name + '&' + mbid_type + '=' + mbid_name

            qry = str1 + str22 + str4
            resp_raw = requests.get(qry)
            resp_dict = resp_raw.json()

    return(resp_dict)


# Should add list for mbids to name

def proc_tags(resp_dict, mbid, mbid_type, c):
    # mbid_name = resp_dict['toptags']['@attr'][mbid_type]
    # name = resp_dict['toptags']['@attr']['artist']

    tags = []
    for i in resp_dict['toptags']['tag']:
        # tag_weight_pr = (i['name'],i['count'])
        tags.append((mbid+"-"+i['name'], mbid, i['name'], i['count'], mbid_type))

        # artist name in quotation marks as otherwise commas would break csv
        # tags can't have commas afaik

    # print(len(tags))
    c.executemany("INSERT OR IGNORE INTO tags (link, mbid, tag, weight, mbid_type) VALUES (?, ?, ?, ?, ?)", tags)
    


if __name__ == '__main__':
    # given a list of mbids and type, the script first checks for which tags are already downloaded, then downloads tags for the rest

    parser = argparse.ArgumentParser()
    parser.add_argument('mbid_type', help='type of mbid: one of track, album, artist')
    parser.add_argument('mbid_file', help='file with mbids')
    parser.add_argument('tag_sqldb', help='output file')

    args = parser.parse_args()

    mbid_type = args.mbid_type
    mbid_file = args.mbid_file
    tag_sqldb = args.tag_sqldb

    tag_sqldb="/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/alb_tags1.sqlite"
    conn = sqlite3.connect(tag_sqldb)
    c=conn.cursor()

    mbids_done = check_existing(c)
    mbids_todo = load_todo(mbid_file)
    
    mbids_todo2 = list(set(mbids_todo) - set(mbids_done))
    print(len(mbids_todo2))

    cntr = 0
    for i in mbids_todo2:

        # c=conn.cursor()
        print(i)
        try:
            resp_dict=get_tags(i, mbid_type, c)
            if check_resp(resp_dict, i) == 1:
                proc_tags(resp_dict, i, mbid_type, c)


            if cntr ==25:
                print(mbids_todo2.index(i), mbids_todo2.index(i)/len(mbids_todo2) )
                conn.commit()
                cntr = 0
            cntr+=1
        except:
            print("mega oopsie!")
            pass
        
conn.commit()

# maybe file just broken by now

# parameters to add:
# - type: artist, album, song
# - infile: list of mbid
# - tagfile: result file

# could add a class to configure mbid type etc just once, but i don't think it wastes much resources to assign them again

# maybe add list of failed ones
