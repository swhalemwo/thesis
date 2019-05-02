import csv
import requests
import argparse


def check_existing(c):
    q=c.execute('SELECT DISTINCT mbid from tags').fetchall()
    f=c.execute('SELECT mbid from failed').fetchall()

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


# def get_tags(mbid, mbid_type, tag_file):
def get_tags(mbid, mbid_type, tag_file):

    # rage against the machine: artist, album, song
    # mbid = '3798b104-01cb-484c-a3b0-56adc6399b80'
    mbid = '10cf6ada-ef3f-3cd3-9a7b-bdc15db2dddf'
    # mbid = 'fa2e0148-835d-473a-babf-c2a3188879d2'

    # mbid_type = 'artist'
    mbid_type = 'album'
    # mbid_type = 'track'

    if mbid_type == 'artist':
        str1="http://ws.audioscrobbler.com/2.0/?method="
        str2=mbid_type +".getTopTags&mbid="
        str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"


        # artst = '936db8ea-2d4b-4d2b-9665-ad8fdecac835'
        qry = str1 + str2 + mbid + str4

        resp_raw = requests.get(qry)
        resp_dict = resp_raw.json()


        # NEED TO FINISH
        # GET NAME, ALBUM from info dict
        # put into get tags call

    else:
        # get info first
        str1="http://ws.audioscrobbler.com/2.0/?method="
        str2=mbid_type + ".getInfo&mbid="    
        str4="&api_key=607aa0a70e1958439a7de088b66a4561&format=json"

        qry = str1 + str2 + mbid + str4
        resp_raw = requests.get(qry)
        resp_dict = resp_raw.json()

        if list(resp_dict.keys())[0] == 'error':
            print(resp_dict['error'], resp_dict['message'])

        else:
            mbid_name = resp_dict[mbid_type]['name']

            if mbid_type=='album':
                artst_name = resp_dict['album']['artist']                

            if mbid_type=='track':
                artst_name = resp_dict['track']['artist']['name']


            str22 = mbid_type + ".getTopTags&artist=" + artst_name + '&' + mbid_type + '=' + mbid_name

            qry = str1 + str22 + str4
            resp_raw = requests.get(qry)
            resp_dict = resp_raw.json()

    if list(resp_dict.keys())[0] == 'error':
        print('oopsie woopsie!')

    else:
        mbid_name = resp_dict['toptags']['@attr'][mbid_type]
        # name = resp_dict['toptags']['@attr']['artist']

        # tags = [mbid,mbid_type,mbid_name]
        # for i in resp_dict['toptags']['tag']:
        #     # tag_weight_pr = (i['name'],i['count'])
        #     tags.append(i['name'])
        #     tags.append(i['count'])

        tags = []
        for i in resp_dict['toptags']['tag']:
            # tag_weight_pr = (i['name'],i['count'])
            tags.append((mbid+"-"+i['name'], mbid, i['name'], i['count'], mbid_type))

            # artist name in quotation marks as otherwise commas would break csv
            # tags can't have commas afaik
            # '/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tags.csv'

        # with open(tag_file, 'a') as fo:
        #     wr = csv.writer(fo)
        #     wr.writerow(tags)
        c.executemany("INSERT OR IGNORE INTO tags (link, mbid, tag, weight) VALUES (?, ?, ?, ?, ?)", tags)
        conn.commit()


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

    mbids_done = check_existing(tag_file)
    mbids_todo = load_todo(mbid_file)
    
    mbids_todo2 = list(set(mbids_todo) - set(mbids_done))
    print(len(mbids_todo2))


    cntr = 0
    for i in mbids_todo2:

        print(i)
        get_tags(i, mbid_type, tag_file)


        if cntr ==25:
            print(mbids_todo2.index(i), mbids_todo2.index(i)/len(mbids_todo2) )
            cntr = 0
        cntr+=1

# parameters to add:
# - type: artist, album, song
# - infile: list of mbid
# - tagfile: result file

# could add a class to configure mbid type etc just once, but i don't think it wastes much resources to assign them again

# maybe add list of failed ones
