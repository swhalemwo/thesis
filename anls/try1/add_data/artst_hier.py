from clickhouse_driver import Client
import time
from discodb import DiscoDB, Q

rows_tags = client.execute("""select mbid, tag, rel_weight, cnt from
(select * from 
    (select * from tag_sums
        join (select tag, count(*) as tag_ap from tag_sums where weight > 10 and rel_weight > 0.05
            group by tag having tag_ap > 50) using tag)
    join (select lfm_id as mbid from acstb) using mbid)
join (select mbid, cnt from song_info3 where cnt > 400 ) using mbid  
where weight > 10 and rel_weight > 0.05""")

df_tags = pd.DataFrame(rows_tags, columns=['lfm_id', 'tag', 'rel_weight', 'cnt'])

songs_tags = df_tags['lfm_id']
len(np.unique(songs_tags))

gnr_song_dict = {}

for r in df_tags.itertuples():
    gnr = r.tag
    
    if gnr in gnr_song_dict.keys():
        gnr_song_dict[gnr].append(r.lfm_id)
    else:
        gnr_song_dict[gnr] = [r.lfm_id]

unq_tags = list(np.unique(df_tags['tag']))

gnr_song_dict['genres'] = unq_tags

db = DiscoDB(gnr_song_dict)

tag_cnts = [len(gnr_song_dict[i]) for i in unq_tags]
tags_srtd=[x for _,x in sorted(zip(tag_cnts, unq_tags), reverse=True)]

tags_srt_sub = tags_srtd[0:200]

def ovlps_mp(gnrs):
    """multiprocessed genre overlap"""
    lens_ttls = []
    for g in gnrs:
        g_lens = []
        # 'genres' has to be set in (disco)db, gnrs is just the subset of genres to be processed by each instance
        # print('genre below')
        print(g)
        # print('genre above')        
        for k,v in db.metaquery(g + '& *genres'):
            x = len(v)
            g_lens.append(x)
        lens_ttls.append(g_lens)
    return(lens_ttls)

inst_sz = 50

mp_input = []

for i in range(4):
    mp_input.append([])

c = 0
for i in tags_srt_sub:
    bin = c % 4
    c+=1
    mp_input[bin].append(i)

p = Pool(processes=4)

t1=time.time()
data = p.map(ovlps_mp, [i for i in mp_input])
t2=time.time()





# * plotting of graph


subsets = np.where(ovlp_ar > 0.75)

subsets_rl = []
c =0
for i in subsets[0]:
    i2 = subsets[1][c]

    if i != i2:
        subsets_rl.append([i,i2])
        
    c+=1

gnr_edges = []
for i in subsets_rl:
    print(tags_srt_sub[i[0]], '---', tags_srt_sub[i[1]])
    gnr_edges.append([tags_srt_sub[i[1]], tags_srt_sub[i[0]]])

from graph_tool.all import *
g = Graph(directed=True)
gt_lbls = g.add_edge_list(gnr_edges, hashed=True, string_vals=True)

gt_lbls_plot = g.new_vertex_property('string')

for v in g.vertices():
    x = gt_lbls[v]
    
    gt_lbls_plot[v] = x.replace(" ", "\n")
    

size = g.degree_property_map('in')

size_scl=graph_tool.draw.prop_to_size(size, mi=6, ma=15, log=False, power=0.5)
size_scl2=graph_tool.draw.prop_to_size(size, mi=10, ma=100, log=False, power=0.5)


graph_draw(g, output_size= (3000,3000),
           output = 'gnr_space.pdf',
           vertex_text = gt_lbls_plot,
           vertex_size = size_scl2,
           vertex_font_size=size_scl
)


# plot hierarchically 
gvd = graphviz_draw(g, size = (20,20),
                    layout = 'dot',
                    vprops = {'xlabel':gt_lbls_plot, 'fontsize':80, 'height':0.03,
                              'width':0.03, 'shape':'point'},
                    # returngv==True,
                    output = 'gnr_space2.pdf')



gvd = graphviz_draw(g, size = (20,20),
                    # layout = 'sfdp',
                    # overlap = 'scalexy',
                    overlap = 'false',
                    vprops = {'xlabel':gt_lbls_plot, 'fontsize':size_scl, 'height':0.03,
                              'width':0.03, 'shape':'point'},
                    eprops = {'arrowhead':'vee', 'color':'grey'},
                    # returngv==True,
                    output = 'gnr_space3.pdf')







# multiprocessing application

# gnrs1 = tags_srt_sub[0:100]
# gnrs2 = tags_srt_sub[100:200]
# gnrs3 = tags_srt_sub[200:300]
# gnrs4 = tags_srt_sub[300:400]

# pool_gnrs = [gnrs1, gnrs2, gnrs3, gnrs4]




# jfc 20k/sec
# now i just have to hope that doesn't go down the drain much when increasing sets

# does it make theoretically sense?
# especially for all the metal sub genres
# should be possible to test: there were like 325 relevant edges from 250k possible
# can easily test if label of superordinate is passed down

# but has to be other way around: if black metal artists are not tagged as metal, but assumed to be metal because it's in the name, they won't be subset
# do i have to do all with REs?

# could do multiple criteria of sub-relation:
# - tag co-occurence
# - label inheritance
# - subset of musical feature space




# * testing of discodb
data = {'mammals': ['cow', 'dog', 'cat', 'whale'],
        'pets': ['dog', 'cat', 'goldfish'],
        'aquatic': ['goldfish', 'whale']}

dbx = DiscoDB(data) # create an immutable discodb object
x = dbx.query(Q.parse('mammals & aquatic'))
[print(i) for i in x]

qry = Q.parse('mammals')

qry2 = Q.metaquery(dbx, 'mammals')

x = dbx.metaquery('pets', qry2)
dbx.

d = DiscoDB({'A': ['B', 'C'], 'B': 'D', 'C': 'E', 'D': 'F', 'E': 'G'})
sorted(d.query(Q.parse('A')))
sorted(d.query(Q.parse('B')))
sorted(d.query(Q.parse('*A')))
sorted(d.query(Q.parse('A | B')))
sorted(d.query(Q.parse('*A | B')))
sorted(d.query(Q.parse('**A | *B')))

mqry = Q.parse('A')
x =
for k,vs in d.metaquery('A | B'):
    print(k, 'wololo', list(vs))
    print(k,v)

for i in x:
    print(i)

    d.metaquery(x)
    
dbx.metaquery([(Q.parse('A'), ['B', 'C'])])


## ** see if metaqueries can be speed up
# yeeees
metals = ['metal','true metal', 'speed metal', 'epic metal', 'sludge metal', 'Nu-metal', 'Iron Maiden', 'finnish metal', 'heavy metal', 'metallica', 'death metal', 'viking metal', 'christian metal', 'trash metal']

gnr_song_dict2 = {}

for i in metals:
    gnr_song_dict2[i] = gnr_song_dict[i]

gnr_song_dict2['metals'] = metals

dbm = DiscoDB(gnr_song_dict2)

m = dbm.metaquery('true metal')

for k,v in dbm.metaquery('true metal & *metals'):
    # print(len(k))
    print(len(v))

[print(i) for i in dbm.query(Q.parse('metals'))]


# *** time comparison with metaqueries

t1 = time.time()
lens = []
for m1 in metals:
    for m2 in metals:
        x = dbm.query(Q.parse(m1 + " & " + m2))
        lens.append(len(x))
t2 = time.time()


t1 = time.time()
lens2 = []
for m1 in metals:
    for k,v in dbm.metaquery(m1 + '& *metals'):
        x = len(v)
        lens2.append(x)
t2 = time.time()
    
# time difference seems to get bigger with more comparisons?
# but not so flexible: have to see in advance that genres have the number of genres i want them to have

# *** try with large dataset

tags_srt_sub = tags_srtd[0:2000]

gnr_dict2 = {}
for i in tags_srt_sub:
    print(i)

    gnr_dict2[i] = gnr_song_dict[i]

gnr_dict2['genres'] = tags_srt_sub

db2 = DiscoDB(gnr_dict2)

t1 = time.time()
lens_ttls = []
for g in tags_srt_sub:
    g_lens = []
    for k,v in db2.metaquery(g + '& *genres'):
        x = len(v)
        g_lens.append(x)
    lens_ttls.append(g_lens)

    print(g)
    
t2 = time.time()
# 500: 28.8 sec: 8.6k/sec
# 1k: 109 sec: 9.1k/sec
# 2k: 421: 9.5/sec
# almost grows? 

#

t1 = time.time()
lens_ttls = []
for g1 in tags_srt_sub:
    g_lens =[]
    for g2 in tags_srt_sub:

        x = db2.query(Q.parse(g1 + " & " + g2))
        g_lens.append(len(x))
    lens_ttls.append(g_lens)
    print(g1)
t2 = time.time()
# 500: 48 sec: 5.2k/sec
# 1k: 182: 5.5k/sec
# 2k: 722: 5.5k/sec


# ** writing/loading

fo = open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/db.disco', 'a')
    db.dump(fo)
    fo.close()

with open('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/db.disco', 'r') as fi:
    dbsx = DiscoDB.load(fi)


# ** multiprocessing theory
from multiprocessing import Process

def f(name):
    print('hello', name)
    for i in range(5):
        print(i)
        time.sleep(1)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

names = ['alice', 'bob', 'caitlin']

for n in names:
    p = Process(target=f, args=(n,))
    p.start()
    p.join()


from multiprocessing import Pool
from time import sleep

def job(num):
    for i in range(1,4):
        print(num, i)
        sleep(1)
    return num * 2

p = Pool(processes=4)
data = p.map(job, [i for i in range(10)])



# looks like metaqueries make it consistenly 66-70% faster
# wonder how it scales with more entries in sets




# ** old comparison function
# def gnr_sub(g1, g2):

#     t1 = time.time()
#     g1_ids = gnr_song_dict[g1]
#     t2 = time.time()

#     g2_ids = gnr_song_dict[g2]
#     t3 = time.time()
#     lack_overlap = set(g1_ids) - set(g2_ids)
#     t4 = time.time()
#     ovlp_prop = 1-(len(lack_overlap)/len(g1_ids))
#     t5 = time.time()
#     return(ovlp_prop)

# t1 = time.time()
# for i in range(1000):
#     gnr_sub('dubstep', 'electronic')
# t2 = time.time()




