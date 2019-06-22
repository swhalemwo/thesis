
# * test  whether multiple edges are accounted for

tedgs = [('1','3'),
         ('1','3'),
         ('1','4'),
         ('1','5'),
         ('2','3'),
         ('2','3'),         
         ('2','6'),
         ('2','7')]

tedgs2 = [('1','3'),
          ('1','4'),
          ('1','5'),
          ('2','3'),
          ('2','6'),
          ('2','7')]


tedgs3 = [('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','4'),
          ('1','5'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','6'),
          ('2','7')]

tedgs4 = [('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','3'),
          ('1','4'),
          ('1','5'),
          ('2','3'),
          ('2','6'),
          ('2','7')]

tedgs5 = [('1','3'),
          ('1','4'),
          ('1','5'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),          
          ('2','6'),
          ('2','7')]

tedgs6 = [('1','3'),
          ('1','4'),
          ('1','5'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),
          ('2','3'),          
          ('2','6'),
          ('2','7'),
          ('6','4'),
          ('1','6'),
          ('3','6'),
          ('7','6'),
          ('2','6')
]



g2 = Graph()
g3 = Graph()
g4 = Graph()
g5 = Graph()
g6 = Graph()
g7 = Graph()


v_nms = g2.add_edge_list(tedgs, hashed=True, string_vals=True)
v_nms2 = g3.add_edge_list(tedgs2, hashed=True, string_vals=True)
v_nms3 = g4.add_edge_list(tedgs3, hashed=True, string_vals=True)
v_nms4 = g5.add_edge_list(tedgs4, hashed=True, string_vals=True)
v_nms5 = g6.add_edge_list(tedgs5, hashed=True, string_vals=True)
v_nms6 = g7.add_edge_list(tedgs6, hashed=True, string_vals=True)
# v = g2.vertex(0)

bs_dir = "/home/johannes/Dropbox/gsss/thesis/anls/try1/"n

g5.set_directed(False)
g6.set_directed(False)
g7.set_directed(False)


graph_draw(g2, vertex_font_size=30, output=bs_dir + 'simt1.pdf', vertex_text = v_nms)
graph_draw(g3, vertex_font_size=30, output=bs_dir + 'simt2.pdf', vertex_text = v_nms2)
graph_draw(g4, vertex_font_size=30, output=bs_dir + 'simt3.pdf', vertex_text = v_nms3)
graph_draw(g5, vertex_font_size=30, output=bs_dir + 'simt4.pdf', vertex_text = v_nms4)
graph_draw(g6, vertex_font_size=30, output=bs_dir + 'simt5.pdf', vertex_text = v_nms5)
graph_draw(g7, vertex_font_size=30, output=bs_dir + 'simt6.pdf', vertex_text = v_nms6)

graph_tool.topology.vertex_similarity(g2, 'jaccard', vertex_pairs = [(0,4)])
graph_tool.topology.vertex_similarity(g3, 'jaccard', vertex_pairs = [(0,4)])
graph_tool.topology.vertex_similarity(g4, 'jaccard', vertex_pairs = [(0,4)])
graph_tool.topology.vertex_similarity(g5, 'jaccard', vertex_pairs = [(0,4)])
graph_tool.topology.vertex_similarity(g6, 'jaccard', vertex_pairs = [(0,4)])
graph_tool.topology.vertex_similarity(g6, 'jaccard', vertex_pairs = [(4,0)])


# jaccard similarity takes multiple edges into account FUCK YEAH

# Any chance to get access to the Million Playlist Dataset? (for science!)


allsims=graph_tool.topology.vertex_similarity(g5, 'jaccard')
allsims2=graph_tool.topology.vertex_similarity(g6, 'jaccard')
allsims3=graph_tool.topology.vertex_similarity(g7, 'jaccard')

allsims3_dice=graph_tool.topology.vertex_similarity(g7, 'dice')

df = pd.DataFrame(allsims)
df2 = pd.DataFrame(allsims2)
df3 = pd.DataFrame(allsims3)
df3_dice = pd.DataFrame(allsims3_dice)


colnames = [v_nms4[i] for i in g5.vertices()]
colnames2 = [v_nms5[i] for i in g6.vertices()]
colnames3 = [v_nms6[i] for i in g7.vertices()]

df.columns = colnames
df2.columns = colnames2
df3.columns = colnames3

# colnames.index(

rnms = {}
for i in colnames:
    rnms[colnames.index(i)] = i

rnms2 = {}
for i in colnames2:
    rnms2[colnames2.index(i)] = i

rnms3 = {}
for i in colnames3:
    rnms3[colnames3.index(i)] = i
    


df2 = df2.rename(index=rnms2)
df3 = df3.rename(index=rnms3)


df.ro

# explanations:
# - order: lower index has to be first -> solvable -> NOPE
# - wrong values seem to be in lower triangle -> NOPE
# - degree: higher degree has to be called first? or last? seems lke it? 
#   works properly when 0 is first, 0 has out degree 8

g7:
(2,3) -> 0.7  -> 7/10
(3,2) -> 0.888 -> 8/9

outdegrees:
- 2: 9
- 3: 8

8/9 would imply that only one is not shared, but that's wrong, boht 1 and 7 are not connected to both
-> 7/10 is correct
--> 2,3 is correct
---> larger one first is correct so far

also compare 4 and 5: no 
might only be issue if one has multiple connections to one node? 

also only if multiple connections are to a node that is also connected to the other node: 6,5 is fine because 2 to which 6 has multiple connections is no neighbor of 5

-> compare 6 and 1: same

- (1, 2): 1.6
- (2, 1): 0.18

-> also node with higher outdeegree (2) has to be first



so far it's always overestimation: can i just take smaller value?
higher outdegree node has to be first because otherwise multiple connections are not considered






# * test again with focus on dice

weights
in which case do they matter
- tag - song: yup
- user - song: multiple edges?
  can i then create extra edges to model it in tag-song relations
-> how are multiple edges considered in dice similarity




xs = [['s1', 'rock'],
      ['s1', 'metal'],
      ['s1', 'metal'],
      ['s1', 'metal'],
      ['s1', 'funk'],
      ['s2', 'funk'],
      ['s2', 'rock'],
      ['s3', 'rock'],
      ['s3', 'metal'],
      ['s4', 'metal'],
      ['s4', 'metal'],
      ['s4', 'metal'],
      ['s4', 'funk'],
      ['s5', 'metal'],
      ['s6', 'funk'],
      ['s7', 'rock'],
      ['s8', 'metal'],
      ['s8', 'funk'],
      ['s9', 'metal'],
      ['s9', 'metal'],
      ['s9', 'funk'],
      ['s10', 'metal'],      
      ['s10', 'funk'],
      ['s10', 'funk'],
      ['s10', 'funk'],
      ['s11', 'funk'],
      ['s11', 'funk'],
      ['s11', 'metal'],
      ['s11', 'metal']]


      

      

gx = Graph(directed=False)
id_mp = gx.add_edge_list(xs, string_vals=True, hashed=True)

# graph_draw(gx,output_size = (150, 70), output = 'debug.pdf',
#            vertex_text = id_mp,
#            vertex_font_size = 4)

graphviz_draw(gx,size = (10, 10), output = 'debug.pdf',
              vprops = {'xlabel': id_mp, 'font_size': 4})

vd = {}
           
for i in gx.vertices():
    vd[id_mp[i]] = int(i)

vertex_similarity(gx, 'dice', [(vd['s4'],vd['s1'])]) 
vertex_similarity(gx, 'jaccard', [(vd['s4'],vd['s1'])])
# s4/s1: 4/5 = 0.8

# adding 2 paths from s4 to metal: 0.57:
# 4/7: 0.57

# numerator:  metal, metal, metal, funk
# denumenator: metal 3x, funk, metal, funk, rock

# add another 2 paths from s1 to metal:
# 0.888, 8/9

# numerator: 2 * (3x metal, funk)
# denuminator: 2 * (3x metal, funk) + rock

vertex_similarity(gx, 'dice', [(vd['s4'],vd['s1'])]) * (gx.vertex(vd['s4']).out_degree() + gx.vertex(vd['s1']).out_degree())/2

# hm seems to be able to account for weights quite well
# shows that 4 are in common 

vertex_similarity(gx, 'dice', [(vd['s1'],vd['s3'])])


# asymmetry weirdness again
vertex_similarity(gx, 'dice', [(vd['s1'],vd['s3'])])
# less problem now: can simply define one direction as true Kappa Kappa Keepo pogchamp
# still, higher degree node has to be first
# same problem here: 
vertex_similarity(gx, 'dice', [(vd['s4'],vd['s3'])])

# can lizardo account for asymmetry?
# does it matter?
# if song 1 is subset of song 2 (all who listen to song 1 listen to song 2 as well), what's the co-occurence?
# how often do they go together?
# like in 100% of cases song 1 goes together with s2, but only in 5% does song 2 go together with s1...

# does bug give me no choice?
# kinda, me thinks
# s3 and s4 only make sense in s4-s3 dir
# and 0.33 makes sense: 
vertex_similarity(gx, 'dice', [(vd['s4'],vd['s3'])])
vertex_similarity(gx, 'dice', [(vd['s9'],vd['s3'])])
vertex_similarity(gx, 'dice', [(vd['s8'],vd['s3'])])

# nonsense
vertex_similarity(gx, 'dice', [(vd['s3'],vd['s4'])])
vertex_similarity(gx, 'dice', [(vd['s3'],vd['s9'])])

# doesn't matter because of symmetry 
vertex_similarity(gx, 'dice', [(vd['s3'],vd['s8'])])

# more unequal degree distribution -> lower similarity 

# what happens when asymmetry refers to different nodes: s10, s9
# s10 has more 
vertex_similarity(gx, 'dice', [(vd['s10'],vd['s9'])])
vertex_similarity(gx, 'dice', [(vd['s9'],vd['s10'])])

# s10: funk*3, metal
# s9: metal*2, funk
# hm would expect 4/7, not 6
# can't see where the 3rd overlap comes from

# NOT GOOD

vertex_similarity(gx, 'dice', [(vd['s1'],vd['s10'])])
vertex_similarity(gx, 'dice', [(vd['s10'],vd['s1'])])

vertex_similarity(gx, 'dice', [(vd['s8'],vd['s9'])])
vertex_similarity(gx, 'dice', [(vd['s9'],vd['s8'])])

vertex_similarity(gx, 'dice', [(vd['s8'],vd['s4'])])
vertex_similarity(gx, 'dice', [(vd['s4'],vd['s8'])])

vertex_similarity(gx, 'dice', [(vd['s10'],vd['s4'])])
vertex_similarity(gx, 'dice', [(vd['s4'],vd['s10'])])

vertex_similarity(gx, 'dice', [(vd['s10'],vd['s9'])])
vertex_similarity(gx, 'dice', [(vd['s9'],vd['s10'])])



# implication: underestimate difference between songs
# differences only register in case of different audience, not audience weight
# weights still work for same song: sim(s8,s4) != sim(s4,s10)

s4/s10 showuld be
s4: metal*3, funk
10: funk*3, metal

intersection is metal, funk
denominator is 8 -> should be 0.5

# how does it look with 2 weighted paths?
vertex_similarity(gx, 'dice', [(vd['s11'],vd['s4'])])
vertex_similarity(gx, 'dice', [(vd['s4'],vd['s11'])], self_loops=False)
# as expected if about same vertices

vertex_similarity(gx, 'dice', [(vd['s11'],vd['s9'])], self_loops=False)
vertex_similarity(gx, 'dice', [(vd['s9'],vd['s11'])], self_loops=False)
# seems to incorporate some weights: when for same ? 

vertex_similarity(gx, 'dice', [(vd['s8'],vd['s11'])])
vertex_similarity(gx, 'dice', [(vd['s11'],vd['s8'])])

vertex_similarity(gx, 'dice', [(vd['s1'],vd['s11'])])
vertex_similarity(gx, 'dice', [(vd['s11'],vd['s1'])])


# other stuff
vertex_similarity(gx, 'dice', [(vd['s9'],vd['s4'])])
vertex_similarity(gx, 'dice', [(vd['s4'],vd['s9'])])

vertex_similarity(gx, 'dice', [(vd['s1'],vd['s10'])]) * (5+4)/2
# no idea how to generalize it
# sometimes degrees get traded?? sometimes not




# ok assume metal and funk are cells:
vertex_similarity(gx, 'dice', [(vd['s11'],vd['s4'])]) * (gx.vertex(vd['s11']).out_degree() + gx.vertex(vd['s4']).out_degree())/2


vertex_similarity(gx, 'dice', [(vd['s4'],vd['s11'])])




st1 = [int(i) for i in gx.vertex(vd['s10']).out_neighbors()]
st2 = [int(i) for i in gx.vertex(vd['s1']).out_neighbors()]

disc_dir = {'st1':st1, 'st2':st2}



set(gx.vertex(vd['s1']).out_neighbors()) & set(gx.vertex(vd['s10']).out_neighbors())
import textdistance
import strsim

vertex_similarity(gx, 'dice', [(vd['s10'],vd['s4'])])
vertex_similarity(gx, 'dice', [(vd['s4'],vd['s10'])])


vertex_similarity(gx, 'inv-log-weight', [(vd['s9'],vd['s4'])])
vertex_similarity(gx, 'inv-log-weight', [(vd['s4'],vd['s9'])])


# * bugreport




# el = [[0,2],
#       [0,2],
#       [1,2],
#       [0,3],
#       [1,3]]

el = [['A', 'v1'],
      ['A', 'v1'],
      ['A', 'v2']]

el = [['A', 'v1'], ['A', 'v1'], ['A', 'v2']]


gx = Graph(directed=False)
id_mp = gx.add_edge_list(el, hashed=True, string_vals=True)

vertex_similarity(gx, 'dice', [(2,1)]) # 1.33
vertex_similarity(gx, 'dice', [(1,2)]) # 0.66



vertex_similarity(gx, 'dice', [(1,2)])
vertex_similarity(gx, 'dice', [(2,1)])

vertex_similarity(gx, 'dice', [(0,1)])
vertex_similarity(gx, 'dice', [(2,0)])


graph_draw(gx,output_size = (100, 100), output = 'debug.pdf', vertex_text=id_mp)



el2 = [['A', 'v1'],
       ['A', 'v1'],
       ['A', 'v1'],
       ['B', 'v1'],
       ['A', 'v2'],
       ['B', 'v2'],
       ['B', 'v2'],
       ['B', 'v2']]


gx2 = Graph(directed=False)
id_mp = gx2.add_edge_list(el2, hashed=True, string_vals=True)

vertex_similarity(gx2, 'dice', [(1,3)])




# * general comparison construction, I/O

# should be possible to skip half of comparisons
# just have to see that i pass right vertex first
# and have to see to read it back in correctly afterwards


sngs = ['s' + str(i) for i in range(1,10)]
gnrs = ['rock', 'metal', 'funk']

# size dict: key vertex id, value (out) degree
sz_dct = {}
for i in sngs:
    sz_dct[int(gx.vertex(vd[i]))] = gx.vertex(vd[i]).out_degree()

comps = []

for i in sngs:
    iv = vd[i]
    # lookups in g more expensive than in dict -> size dict
    isz = sz_dct[iv]
    
    comps2 = []
    for k in sngs:
        kv = vd[k]
        ksz = sz_dct[kv]

        # if isz > ksz:
        comps2.append([iv, kv])
        # else if 

    comps.append(comps2)
        
cmps = list(itertools.chain.from_iterable(comps))

need to check whether already there
store in np array?
# yup lists work fine in array

can make dict with positions? i.e. which position in list links to matrix element? 

ar1 = [[['a','a'], 'B', 'C'], ['d', 'e', 'f'], ['g', 'h', 'i']]
arx = np.array(ar1)

comp_ls = []
for s1 in sngs:
    s1_l = []
    for s2 in sngs:
        s1_l.append([s1, s2])
    comp_ls.append(s1_l)    
    
comp_ar = np.array(comp_ls)

lenx = len(sngs)

t1 = time.time()
# i,k = 4,7
for i in range(lenx):
    for k in range(i, lenx):
        # print(i,k)
        
        vs = comp_ar[i,k]
        v1 = vs[0]
        v2 = vs[1]

        v1_sz = sz_dct[vd[v1]]
        v2_sz = sz_dct[vd[v2]]

        if v1_sz > v2_sz:
            fnl_ord = [v1,v2]
        elif v1_sz < v2_sz:
            fnl_ord = [v2,v1]
        else:
            fnl_ord = [v1,v2]

        # set both sets to same order

        comp_ar[i,k] = fnl_ord
        comp_ar[k,i] = fnl_ord

        # set other side to same order
        # print(k,i)
        # print(comp_ar[k,i])
t2 = time.time()





for i in range(lenx):
    for k in range(lenx):

        vs = comp_ar[i,k]
        sz1 = sz_dct[vd[vs[0]]]
        sz2 = sz_dct[vd[vs[1]]]
        
        print(sz1, sz2, sz2 > sz1, sz1 >= sz2)

        # looking good

        
