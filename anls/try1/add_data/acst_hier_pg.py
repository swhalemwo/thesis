# * playground
# ** hausdorff distance
from scipy.spatial.distance import directed_hausdorff

u = np.array([(1.0, 0.0),
              (0.0, 1.0),
              (-1.0, 0.0),
              (0.0, -1.0)])
u1 = [i[0] for i in u]


v = np.array([(2.0, 0.0),
              (0.0, 2.0),
              (-2.0, 0.0),
              (0.0, -4.0)])

plt.scatter(u[:,0], u[:,1])
plt.scatter(v[:,0], v[:,1])
plt.show()

directed_hausdorff(u,v)
directed_hausdorff(v,u)

HD is largest of shortest distances?
for each point get the shortest distance to other set, and then take the largest of those? 
seems to be

i think asymmetry is fine since unit of analysis is genre

seems to be sensititve to outliers tho (wikipedia figure)
https://en.wikipedia.org/wiki/Hausdorff_distance

PIazzai use mean of min minimum distances



# complete not the same:
# low fractions of x2 (0.01), if x1 fraction is high (0.2):
# overall fraction is 20
# log is not that high but still very high
# does not happen if i drop it

# KLD only works if for all x where x2 (qx) is 0, x1 (px) is also 0
# only works for sub-concept relation
# absolute continuity: some calculus stuff


# can KLD account for unequal likelihoods?
# does sum of 1 pose problem?
# at all levels, X more likely than Y
# seems to ask: given you have a object of distribution X, you likely is it to be in place Z
# not: given that you're in place Z, how likely are you to be part of X or Y?

# do i need it? could use it for sub-concept , if subset relation exists

# but would mean i need another measure for non subsets

# maybe makes sense: asking how similar are swimmers to athlethes is a different question than asking how similar are scientists to athletes

# Hannan also want to use KDV for cohort distinctiveness
# requires non-zero values on all features
# idk
# should not work if subconcepts are more specific, have not all the same dimensions
# swimmer (subconcept of athlete) has different dimensions than bodybuilder (lifts weight)
# probability distributions in those two dimensions are not overlapping

# Hannan use cosine similarity (p.91), then distances (exponential) 
# but that's symmetric
# maybe that bad in cohort tho: need to see volume distribution in cohorts
# also problem that sub-sub concepts (lowest level) will show up as members of higher level
# exclude if (next to being subconcept) it is also a subconcept of another subconcept

# cosine similarity: also doesn't take probability distribution into account
# first waste of information
# second 



klbk_lblr_dist(x2,x1)


# ** cosine similarity
# might not even have to normalize for it, but won't really distort much me thinks

from sklearn.metrics.pairwise import cosine_similarity

x= cosine_similarity(acst_mat)


plt.hist(x[np.where(0<np.tril(x))], bins='auto')
plt.show()


plt.hist(x[(np.tril(x) > 0) & (np.tril(x) < 1)], bins='auto')
plt.show()

from scipy.spatial import distance
distance.euclidean

x2 = sklearn.metrics.pairwise.euclidean_distances(acst_mat)
plt.hist(x2[np.tril(x2) > 0], bins='auto')
plt.show()

# what's the point of putting it into network really
# -> need to functionalize the network generation
# but more the relevant feature extraction -> straightforward to compare

# ** calculating overlap/divergence

# need to find cells where c2 is nonzero and c1 is zero

emp_cells = np.logical_and(c2.spc_fl > 0, c1.spc_fl== 0).nonzero()

emp_rows = [[0 for i in range(100)] for i in range(100)]
emp_spc = np.array(emp_rows)

for i in range(len(emp_cells[1])):
    emp_spc[emp_cells[0][i],emp_cells[1][i]]=c2.spc_fl[emp_cells[0][i],emp_cells[1][i]]

plt.imshow(emp_spc, interpolation='nearest')
plt.show()

pct_ncvrd = sum(sum(emp_spc))/sum(sum(c2.spc_fl))

c2_spc = c2.spc_fl

for i in range(len(emp_cells[1])):
    c2_spc[emp_cells[0][i], emp_cells[1][i]] = 0


xx = list(itertools.chain.from_iterable(c2_spc))
yy = list(itertools.chain.from_iterable(c1.spc_fl))

entropy(xx, yy)

yy2 = c1.h1 + c1.h2
xx2 = c2.h1 + c2.h2

entropy(xx2, yy2)


xs2 = [i for i in range(len(yy2))]
ax = plt.axes()
ax.plot(xs2, xx2)
ax.plot(xs2, yy2)
plt.show()



# ** KLD1
# *** normal kld


kld_el = sbst_eler(ar_cb, operator.lt, 0.12)

# kld_rel = np.where(np.array(klds) < 0.05)
# kld_el = np.array(kld_cmps)[kld_rel[0]]


g_kld = Graph()
g_kld_id = g_kld.add_edge_list(kld_el, hashed=True, string_vals=True)

graph_pltr(g_kld, g_kld_id, 'acst_spc4.pdf')

x = set(g_kld_id[i] for i in g_kld.vertices())

vd_kld, vd_kld_rv = vd_fer(g_kld, g_kld_id)

[print(g_kld_id[i]) for i in g_kld.vertex(vd_kld['indie']).in_neighbors()]

graph_draw(g_kld, output='g_kld.pdf')
# amount of reciprocal relationships?
# not if i just get 3 most influential parents or so

# *** test how much thrshold has to be relaxed to get most/all genres included
# quite alot; to the extent that most genres will have at least dozes of superordiates and suporidates
# -> how to 

gnr_cnt = []

for i in np.arange(0.01, 0.25, 0.0025):
    kld_el = sbst_eler(ar_cb, operator.lt, i)

    g_kld = Graph()
    g_kld_id = g_kld.add_edge_list(kld_el, hashed=True, string_vals=True)

    # print(g_kld)
    # print(i)

    gnr_cnt.append(len(list(g_kld.vertices())))

xs = np.arange(0.01, 0.25, 0.0025)
fig = plt.figure()
ax = plt.axes()
ax.plot(xs, gnr_cnt)
plt.show()


# *** explore kld measures:
case:
gnr = 'Death Doom Metal'
gnr_nbrs = [g_kld_id[i] for i in list(g_kld.vertex(vd_kld['Death Doom Metal']).out_neighbors())]
gnr_id = gnrs.index(gnr)

klds = kld_mp([gnr])

klds2 = [i for i in klds[0] if i < 0.05]
plt.hist(klds2, bins='auto')
plt.show()

for k in gnr_nbrs:
    i_v = acst_mat[gnrs.index(gnr)]
    k_v = acst_mat[gnrs.index(k)]

    b_zeros = np.where(k_v==0)
    a_sum_b_zeros = sum(i_v[b_zeros])
    prop_missing = a_sum_b_zeros/sum(i_v)
            
    if prop_missing == 0:
        ent = round(entropy(i_v, k_v),3)
        print(gnr, k, ent, 'complete')
                
    elif prop_missing < 0.05:
                
        i_v2 = np.delete(i_v, b_zeros)
        k_v2 = np.delete(k_v, b_zeros)

        ent = round(entropy(i_v2, k_v2),3)
        print(gnr, k, ent, 'incomplete', prop_missing)

x = [i for i in range(0,120,1)]

fig = plt.figure()
ax = plt.axes()
# ax.plot(x[0:10], i_v[0:10])
ax.plot(x[0:10], k_v[0:10])
plt.show()

a1,a2,nsns = plt.hist(acst_gnr_dict[gnr]['dncblt'], bins=10)
plt.hist(acst_gnr_dict[k]['dncblt'], bins=10)
plt.show()

fig = plt.figure()
ax = plt.axes()
ax.plot(x[0:10], a1/sum(a1))
# ax.plot(x[0:10], i_v[0:10])
plt.show()

# ** debug w_std2: should not result in symmetric similarities
# similarites are symmetric, but have to be processed


asym_sim_ar = asym_sim(gnrs, vd)
nph(asym_sim_ar)

el_asym = sbst_eler(asym_sim_ar, 0.99)
# this seems to put super general things (10 of 10 stars) as super wide
# wonder if i should use the reverse distance (superordinate to subordinate) somehow, might have relevant information

g_asym = Graph()
asym_weit = g_asym.new_edge_property('double')

asym_id = g_asym.add_edge_list(el_asym, hashed=True, string_vals=True, eprops = [asym_weit])

vd_asym, vd_asym_rv = vd_fer(g_asym, asym_id)

graph_pltr(g_asym, asym_id, 'acst_spc4.pdf')

for v in g_asym.vertices():
    print(asym_id[v], v.in_degree())

# x = np.reshape(asym_sim_ar, (1, len(gnrs)**2))
# plt.hist(x[0], bins='auto')
# plt.show()

# is not asymmetric now
# that's what happens when you standardize 
# could standardize so that max = 1,
# 3,3,2 -> 1,1,0.6
# 3,0.5,0.5 -> 1,1,0.1666
# scaling down with max, not total sum -> ask MARIEKE


# you imbecil
# you fucking moron
# when you get directed similarity you have to divide it by the bases
# first get common stuff
# then divide that by each vertex' ttl 

# fixed but still seems not too good: some pointless genres (00s, 10 stars become super large)


# * old KLD function
def kld_mp(chnk):
    """multiprocessing function for KLD"""
    
    ents_ttl = []

    for gnr in chnk:

        i_id = gnr_ind[gnr]
        i_v = acst_mat[i_id]
        gnr_ents = []
        
        for k in gnrs:
            
            k_id = gnr_ind[k]
            k_v = acst_mat[k_id]
            
            b_zeros = np.where(k_v==0)
            a_sum_b_zeros = sum(i_v[b_zeros])
            prop_missing = a_sum_b_zeros/sum(i_v)
            
            if prop_missing == 0:
                ent = entropy(i_v, k_v)
                
            elif prop_missing < 0.05:
                
                i_v2 = np.delete(i_v, b_zeros)
                k_v2 = np.delete(k_v, b_zeros)

                ent = entropy(i_v2, k_v2)
            else:
                ent = math.inf

            gnr_ents.append(ent)
        ents_ttl.append(gnr_ents)
    return(ents_ttl)

# ** binarizing graph to get asymmetry: BAD (at least for hierarchical links) because requires deleting weights and then creates nonsense

acst_mat_bn = np.zeros(acst_mat.shape)
acst_mat_bn[np.where(acst_mat > 0.3)] = 1

sums = np.sum(acst_mat_bn, axis=1)
nph(sums)

vrbl_nd_strs_raw = [[vrbl + str(i) for i in range(1,11)] for vrbl in vrbls]
vrbl_nd_strs = list(itertools.chain.from_iterable(vrbl_nd_strs_raw))

el_bin = []

for gnr in gnrs:
    gnr_ar_bin = acst_mat_bn[gnr_ind[gnr]]
    ftrs_bin = np.array(vrbl_nd_strs)[np.where(gnr_ar_bin == 1)]

    gnrs_el_bin = [(gnr, f) for f in ftrs_bin]
    el_bin = el_bin + gnrs_el_bin

g_bin = Graph()
g_bin_id = g_bin.add_edge_list(el_bin, hashed=True, string_vals=True)

vd_bin, vd_bin_rv = vd_fer(g_bin, g_bin_id)

cmps = all_cmps_crubgs(gnrs, vd_bin, 'product')

sims = vertex_similarity(g_bin, 'dice', vertex_pairs = cmps)

ovlp_ar = asym_sim(g_bin, gnrs, vd_bin)

# shows that high overlap doens't mean high similarity?
# high overlap doesn't mean high similarity because the similarity here is symmetric
# if there isn't much overlap for one genre, but super much for the other, for example i think

nph(sims_ar[np.where(ovlp_ar > 0.8)])
nph(ovlp_ar[np.where(ovlp_ar > 0.8)])

# subsetting with absolute stuff no good
bin_el1 = sbst_eler(ovlp_ar, operator.gt, 0.9)

bin_el2 = kld_n_prnts(1-ovlp_ar ,4)

g_hr_bin = Graph()
g_hr_bin_sim = g_hr_bin.new_edge_property('float')
g_hr_bin_id = g_hr_bin.add_edge_list(bin_el2, hashed = True, string_vals=True, eprops = [g_hr_bin_sim])

graph_pltr(g_hr_bin, g_hr_bin_id, 'acst_spc7.pdf', 1)

vd_hr_bin, vd_hr_bin_rv = vd_fer(g_hr_bin, g_hr_bin_id)

## not exactly sure if that works: should rewrite kld_n_prnts into general el function similar to sbst_eler


# debug because looks weird
# pink floyd & melancholic looks ok: melancholic is sub-genre of pink floyd, has 21 of the 24 features

# also same values with acoustic guitar as superset of jazz guitar
# check further down


g1 = "Pink Floyd"
g2 = 'melancholic'

g1 = 'acoustic guitar'
g2 = 'jazz guitar'


pr = [vd_bin[g1],vd_bin[g2]]

(vertex_similarity(g_bin, 'dice', vertex_pairs = [pr]) * (g_bin.vertex(vd_bin[g1]).out_degree() + g_bin.vertex(vd_bin[g2]).out_degree())/2) / g_bin.vertex(vd_bin[g1]).out_degree()


g_hr_bin_sim[g_hr_bin.edge(g_hr_bin.vertex(vd_hr_bin[g1]), g_hr_bin.vertex(vd_hr_bin[g2]))]
g_hr_bin_sim[g_hr_bin.edge(g_hr_bin.vertex(vd_hr_bin[g1]), g_hr_bin.vertex(vd_hr_bin[g2]))]

cmon_ftrs = set(g_bin.vertex(vd_bin[g1]).out_neighbors()) & set(g_bin.vertex(vd_bin[g2]).out_neighbors())
g1_ftrs = set(g_bin.vertex(vd_bin[g1]).out_neighbors()) - set(g_bin.vertex(vd_bin[g2]).out_neighbors())
g2_ftrs = set(g_bin.vertex(vd_bin[g2]).out_neighbors()) - set(g_bin.vertex(vd_bin[g1]).out_neighbors())
[print(g_bin_id[i]) for i in cmon_ftrs]
[print(g_bin_id[i]) for i in g1_ftrs]

# so far makes sense technically as in doesn't seem like errors
# but not really thematically
# maybe really different threshold? 
# 0.3 threshold: suomi is now most central WTFFF
# just has 2 features -> so many stuff is super similar to it

# but the lower the threshold, the less asymmetry
# 0.03: still enough asymmetry, but nonsense stuff like "Battlestar Galactica" and "j dilla" now eats up everything

# hmm both have among the lowest attributes possible
# -> they are similar to a lot of other ones
# they are basically super general now

g1 = 'Battlestar Galactica'
g2 = 'j dilla'
g3 = 'Suomi'
len(set(g_bin.vertex(vd_bin[g3]).out_neighbors()))

# hm it's kinda i'm throwing away information, the more the higher the threshold
# tbh i dont' know if it's salvageable
# to geth this asymmetric similarity i have to binarize
# and to binarize means throwing away information on weights
# but weights are exactly the thing that makes it work in the first place
-> BINARIZING BAD, rather find different measure
# at least for hierarical linkage

# could also still weigh, 

g_asym, asym_sim, g_asym_id, vd_asym, vd_asym_rv = kld_proc(bin_el2)
graph_pltr(g_asym, g_asym_id, 'acst_spc6.pdf', 1)


# should try with different thresholds (0.1, 0.15, 0.2) and see if difference

# ** song similiarity

# tx1 = time.time()
# dfcx_proc('electronic')
# tx2 = time.time()


# unq_artsts
# gnr_gini
# avg_age
# age_sd
# nbr_rlss_tprd
# ttl_size
# prop_rls_size
# dist_mean
# dist_sd



# metal genres seem to have skewed or even bimodal distributions
# but also i wonder if the other genres with normal distributions centered between 1 and 1.5 are ok
# basically means nothing is really similar to each other?
# 

# * compare KLDs (test how different they really are to disprove Kotamy (cited by Piazzai))

n1 = [np.random.normal(0, 1) for i in range(100000)]
n2 = [np.random.normal(0, 2) for i in range(100000)]

bins = np.arange(math.floor(min(n2)), math.ceil(max(n2)), 0.1)

# has to be overall 0
n3 = [(i**2)*4 for i in np.arange(-20, 20)]

n3 = [np.absolute((i**3)*0.2) for i in np.arange(-20, 20)]
n3_2 = [i-(sum(n3)/len(n3)) for i in n3]

peak = list(n1_hist).index(max(n1_hist))
n3_cplt = np.histogram(n1, bins = bins)[0]
c = -20
for i in n3_2:
    print(c, peak+c)
    n3_cplt[peak + c] = n3_cplt[peak + c] + i
    c+=1


n1_hist = np.histogram(n1, bins = bins)[0]
n2_hist = np.histogram(n2, bins = bins)[0]

plt.plot(n1_hist)
plt.plot(n2_hist)
plt.plot(n3_cplt)
plt.show()

entropy(n1_hist, n2_hist) # 0.31-0.33

entropy(n1_hist, n3_cplt)
entropy(n3_cplt, n1_hist)

# yup: changes between 
# Komaty BTFOp

# can't really replicate the changes to the normal function, but there is substantial divergence between two normal dists with different SD, much more than claimed 

# * more hierarchy inference

# how is this related to conditional independence?
# i feel it is somehow

# ** residual
# first select most similar
# then see how much others add

gnr = "vocal jazz"
gnr = 'fun metal'
gnr = 'skate punk'
gnr = 'latin pop'
[g_kld2_id[i] for i in g_kld2.vertex(vd_kld2[gnr]).in_neighbors()]

prnts = list(g_kld2.vertex(vd_kld2[gnr]).in_neighbors())
prnt_edges = list(g_kld2.vertex(vd_kld2[gnr]).in_edges())

[kld_sim[i] for i in prnt_edges]

prntv = prnt_edges[[kld_sim[i] for i in prnt_edges].index(min([kld_sim[i] for i in prnt_edges]))].source()
prnt1 = g_kld2_id[prntv]

gnr_id = gnr_ind[gnr]
prnt1_id = gnr_ind[g_kld2_id[prntv]]

gnr_dist = acst_mat[gnr_id]
prnt1_dist = acst_mat[prnt1_id]

entropy(gnr_dist, prnt1_dist)
nph(ar_cb[gnr_id][np.where(ar_cb[gnr_id] < math.inf)])
npl([gnr_dist[0:50], prnt1_dist[0:50]], m=True)

# can't be minus: probability can't be negative
# res1 = gnr_dist - prnt1_dist
# npl(res1[0:20])

res2_1 = gnr_dist/prnt1_dist
res2_2 = [-math.log(i) for i in res2_1]
# res2_2 = prnt1_dist/gnr_dist
npl(res2_1[0:50])
npl(res2_2[0:50])

npl([res2_1[0:50], gnr_dist[0:50], prnt1_dist[0:50]], m=True)
# hm i want stuff in vocal jazz that is not explained by jazz
# res_2_1 seems better: is high at start where vocal jazz higher than jazz -> stuff should be added here
# res2_2 makes also no sense because gnr_dist can be 0 for values of prnt1 dist, but not other way around
npl(res2_1[0:40])

res3 = entropy_wot_sum(gnr_dist, prnt1_dist)
npl(res3[0:140])
# prnt2_dist = acst_mat[gnr_ind[g_kld2_id[prnts[1]]]]

# entropy(res2_1, prnt2_dist)
# prnt2 doesn't seem to add much


simx = entropy(res2_1[None,:].T[:,:,None], acst_mat.T[:,None,:])
simx2 = entropy(res3[None,:].T[:,:,None], acst_mat.T[:,None,:])

simx3 = entropy(res3[None,np.where(res3 >= 0)].T[:,:], acst_mat[:,np.where(res3 >= 0)].T[:])
nph(simx3[np.where(simx3 < math.inf)])
# variable dumping might work?

acst_mat.T[np.where(res3 >= 0),None,:])

acst_mat[:,np.where(res3 >= 0)].T[:].shape


gnrs[list(simx[0]).index(min(simx[0]))]
nph(simx[0][np.where(simx[0] < math.inf)])


gnrs[list(simx2[0]).index(min(simx2[0]))]
nph(simx2[0][np.where(simx2[0] < math.inf)])


def entropy_wot_sum(pk, qk=None, base=None):
    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=0)
    if qk is None:
        vec = scipy.special.entr(pk)
    else:
        qk = np.asarray(qk)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0*qk / np.sum(qk, axis=0)
        vec = scipy.special.rel_entr(pk, qk)
    return(vec)

# vocal jazz
# hm that's kinda unlikely
# could make maximum of first ones threshold
# there is no meaningful relationship tho

# fun metal
# jesus that's alone everything is fucking distant


# does the residual make sense? 
# reconsider subtraction: maybe just get ride of everything negative, just focus on features not explained?
# and ignore features overexplained?
# to some extent i'm doing it: what really highlights/jumps out are the features where genre is high and parent is low, can be by a factor of 50 or so

# maybe the individual parts of KLD? 
# can be negative, also have the same thing of large differences


# overall, there seems to be a sense of arbiraryness: creating residuals with the first one have nothing to do wiht the rest anymore
# but the second parent is not much different..
# if i took him first, would it still possible to see whether the actual first one actuall is the first one?

# i don't like the overall distribution of divergences for the distant ones
# there's so much gap, and then relatively close other genres, which basically means a bit measurement error is constructing a different set of parents
# what i would like is like on 0.5, one 0.1 and one 0.15 for fucking all
# and it's not only for genres with low number of songs or artists, same pattern is for latin pop with 80 songs and 41 artist



# ** combinations from the start
# is there a genre in combination with instrumental that has a higher similarity with genre? 
# weights? unclear

# need overall matrix of prnt_dist
prnt_ovrl = np.expand_dims(prnt1_dist, axis = 1)

b = np.repeat(a[:, :, np.newaxis], 3, axis=2)

prnt_ovrl = np.repeat(prnt1_dist[:,np.newaxis], len(gnrs), axis = 1).T
gnr_cmb = acst_mat + prnt_ovrl

gnr_cmb[gnr_ind[gnr]] = [1 for i in range(120)]

simx3 = entropy(gnr_dist[None,:].T[:,:,None], gnr_cmb.T[:,None,:])

min(simx3[0])

add1 = gnrs[list(simx3[0]).index(min(simx3[0]))]
entropy(gnr_dist, prnt1_dist)
entropy(gnr_dist, prnt1_dist + acst_mat[gnr_ind[add1]])

npl([gnr_dist, prnt1_dist + acst_mat[gnr_ind[add1]]/2, prnt1_dist, acst_mat[gnr_ind[add1]]], m = True)

nph(simx3[0][np.where(simx3[0] < math.inf)])

# not good: adds itself: itself + prnt1 is most similar combination
# but others still lower (0.4 instead of 0.58)
# suggested addition is 'blacker than the blackest black times infinity', which is a bit cause it's super atypical too

# maybe just need to try out?

# original similarity 
entropy(gnr_dist, acst_mat[gnr_ind[add1]])
# is infinite.. wtf
# tbh it would kinda work if i don't assume strict subsettedness
# but does that make much sense in terms of dimensions? if superordinate limits range, all sub genres of it should be in that area

# need to see what it means theoretically to add dists together
# also is feature overexplanation really that bad? consider theory, in particular wrt features vs dimensions
# also relate to compositionality
# could question just be: where do underexplained features come from?

# maybe i should reduce number of features..
# there are never ever 10 different steps of danceability..
# probably 5 are still too much but it's in the right direction.. want uneven number and 3 is a bit low, unable to capture nuances
# atm there's 75% inf, which is bad because data loss for pct mean and infering potential genre relations
len(list(ar_cb[np.where(ar_cb == math.inf)]))/len(gnrs)**2
# down to 48%
# i think now i can really expect a subgenre to be in the same space as a main genre
# at least 30 songs, which have to hit at most 5 cells
# not quite sure but it should be around 0.8**30


# if i go down that route there's no reason not to do all 2 combinations from the start
# 480k combinations, fucking amazing

# ** regression

from sklearn.linear_model import LinearRegression


# *** single case
g1 = 'witch house'
g2 = 'House'

g1_vlus = acst_mat[gnr_ind[g1]].reshape(-1, 1)
g2_vlus = acst_mat[gnr_ind[g2]]

model = LinearRegression()

model.fit(g1_vlus, g2_vlus)

model.score(g1_vlus, g2_vlus)
model.intercept_
model.coef_


# *** multiple vars
gnrs = ['pop', 'House', 'rock', 'rap']
preds = [gnr_ind[i] for i in gnrs]
pred_mat = acst_mat[preds].T

reg = LinearRegression()
reg.fit(g1_vlus, pred_mat)

reg.coef_

reg.score(g1_vlus, pred_mat)

y_pred = model.predict(g1_vlus)
r2 = sklearn.metrics.r2_score(g1_vlus, y_pred)



# *** lasso


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
#print cancer.keys()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
#print cancer_df.head(3)
X = cancer.data
Y = cancer.target

# X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0, random_state=31)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train,y_train)

# lasso.fit(X,Y)

train_score=lasso.score(X_train,y_train)

test_score=lasso.score(X_test,y_test)

coeff_used = np.sum(lasso.coef_!=0)
print(coeff_used)

# using the entire dataset to train means only two features are selected
#


# *** lasso tags

lasso_el = []
las_scrs = []

for gnr in gnrs:

    gnr_vlus = acst_mat[gnr_ind[gnr]]
    preds2 = np.delete(acst_mat, gnr_ind[gnr], axis=0).T

    gnrs_pred = np.delete(np.array(gnrs), gnr_ind[gnr])

    lasso = Lasso(0.016)
    lasso.fit(preds2,gnr_vlus)
    coef_used = np.where(lasso.coef_!=0)
    print(len(coef_used[0]))

    # [print(gnrs_pred[i], lasso.coef_[i]) for i in coef_used[0]]
    # hm that's kinda weird parents sometimes
    # need to test overall tho

    las_scrs.append(lasso.score(preds2, gnr_vlus))

    gnr_el = []
    for i in coef_used[0]:
        prnt_pr = (gnrs_pred[i], gnr, lasso.coef_[i])
        gnr_el.append(prnt_pr)

    lasso_el = lasso_el + gnr_el
    

g_las = Graph()
g_las_waet = g_las.new_edge_property('float')

g_las_id = g_las.add_edge_list(lasso_el, string_vals = True, hashed = True, eprops = [g_las_waet])
g_las_vd, g_las_vd_rv = vd_fer(g_las, g_las_id)

graph_pltr(g_las, g_las_id, 'lasso_spc1.pdf', 1)

# hmmm
# rock all over the place
# metal, punk, electro have their own areas kinda
# who the fuck is j_dilla: some rapper
# maybe introduce constraint that genre is not allowed to be more than X (50/60%) of one artist
# actually why don't i just hard-remove any artists? 

# also wtf is the deal with aggressive and german lyrics
# and fucking American Idol?

# also general genres (rock, metal) are basically irrelevant
# also fuccking "aggressive"


# other stuff: weigh
# - lasso cutoff by size of genre: can't see a substantial reason for it
- predictors by their size
  does that even make sense? usually you weigh rows, not variables
  could multiply the columns by their size but it wouldn't make much sense 
  

# it's a plausible model of constructing tho


# *** lasso weighted
# **** https://stackoverflow.com/questions/44757238/how-to-configure-lasso-regression-to-not-penalize-certain-variables

""" data """
import numpy as np
from sklearn import datasets
diabetes = datasets.load_diabetes()
A = diabetes.data[:150]
y = diabetes.target[:150]
alpha=0.1
weights=np.ones(A.shape[1])

""" sklearn """
from sklearn import linear_model
clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
clf.fit(A, y)

""" scipy """
from scipy.optimize import minimize
def lasso_min(x):  # following sklearn's definition from user-guide!
    return (1. / (2*A.shape[0])) * np.square(np.linalg.norm(A.dot(x) - y, 2)) + alpha * np.linalg.norm(weights*x, 1)

def lasso_min2(x, A, y, weights):
    cp1 = (1. / (2*A.shape[0]))
    cp2 = np.square(np.linalg.norm(A.dot(x) - y, 2)) 
    cp3 = alpha * np.linalg.norm(weights*x, 1)

    res = cp1 * cp2 + cp3
    # print(res)
    return(res)


""" Test with weights = 1 """
x0 = np.zeros(A.shape[1])
res = minimize(lasso_min, x0, method='L-BFGS-B', options={'disp': False})
res2 = minimize(lasso_min2, x0, args=(A, y, weights), method='L-BFGS-B', options={'disp': False})

print('Equal weights')
print(lasso_min(clf.coef_), clf.coef_[:5])
print(lasso_min(res.x), res.x[:5])
print(lasso_min(res.x), res2.x[:5])

""" Test scipy-based with special weights """
weights2 = weights
weights2[[0, 3, 5]] = 0.0
res = minimize(lasso_min, x0, method='L-BFGS-B', options={'disp': False})
res2 = minimize(lasso_min2, x0, args=(A, y, weights2), method='L-BFGS-B', options={'disp': False})
print('Specific weights')
print(lasso_min(res.x), res.x[:5])
print(lasso_min(res2.x), res2.x[:5])

# **** try out
gnr = 'ambient'
gnr = 'death metal'
gnr = 'progressive power metal'


gnr_vlus = acst_mat[gnr_ind[gnr]]
# get 30 most similar genres
# idx = np.argpartition(ar_cb[gnr_ind[gnr]], 30)
idx = np.argpartition(ar_cb[gnr_ind[gnr]], range(30))
prnts = idx[1:30]

gnrs_preds = acst_mat[prnts,:].T

szs = [sz_dict[gnrs[i]] for i in prnts]
szs_log = [math.log(i) for i in szs]
szs_log22 = np.array([i/max(szs_log) for i in szs_log])
szs_rv = [1 - i for i in szs_log22]
szs_xx = [1/sz_dict[gnrs[i]] for i in prnts]
szs_xx2 = [i/max(szs_xx) for i in szs_xx]

# older stuff
# too much to estimate all for now
# szs = [sz_dict[i] for i in gnrs]
# szs_log = [math.log(i) for i in szs]


preds2 = np.delete(acst_mat, gnr_ind[gnr], axis=0).T
gnrs_pred = np.delete(np.array(gnrs), gnr_ind[gnr])
szs2 = np.delete(np.array(szs), gnr_ind[gnr])
szs_log = np.delete(np.array(szs_log), gnr_ind[gnr])
szs_none = np.array([1 for i in szs_log])
szs_log2 = np.array([i/max(szs_log) for i in szs_log])
# maybe szs can't be > 1? 

alpha = 0.016
x0 = np.zeros(preds2.shape[1])


# does not work with equal weights: produces different stuff
res2 = minimize(lasso_min2, x0[0:20], args=(preds2[:,0:20], gnr_vlus, szs_none[0:20]), method='L-BFGS-B', options={'disp': True, 'maxiter':100000})
npl(res2.x)

res2 = minimize(lasso_min2, x0[0:50], args=(preds2[:,0:50], gnr_vlus, szs_log2[0:50]), method='Powell', options={'disp': True})
npl(res2.x)

# using only closest predictors (kld)
res2 = minimize(lasso_min2, x0[0:29], args=(gnrs_preds, gnr_vlus, szs_xx2), method='Powell', options={'disp': True})
npl(res2.x)
[gnrs[prnts[i]] for i in np.where(res2.x>0.1)[0]]
[gnrs[i] for i in prnts]

# also does not work with original function lasso_min, same results as with rewritten function
# uses different optimization function: coordinate descent not available for minimize, Powell looks somewhat similar


x0 = np.zeros(50)
y = gnr_vlus
A = preds2[:,0:20]
weights = [1 for i in x0]
res3 = minimize(lasso_min, x0, method='L-BFGS-B', options={'disp': True})



lasso = Lasso(0.01)
lasso.fit(preds2[:,0:50],gnr_vlus)

lasso.fit(gnrs_preds,gnr_vlus)
npl(lasso.coef_)
[gnrs_pred[i] for i in np.where(lasso.coef_ > 0.01)]
# no overlap at all between using the 30 most similar with weights and everything without
# do i need coordinate descend with weights? f

# **** lasso 2-step

gnr = 'ambient'
gnr = 'Technical Death Metal'

gnr_vlus = acst_mat[gnr_ind[gnr]]
preds2 = np.delete(acst_mat, gnr_ind[gnr], axis=0).T
gnrs_pred = np.delete(np.array(gnrs), gnr_ind[gnr])

lasso = Lasso(0.00001, precompute=True, max_iter = 30000)
lasso.fit(preds2,gnr_vlus)
npl(lasso.coef_)
las_res = [(gnrs_pred[i], lasso.coef_[i]) for i in np.where(lasso.coef_ > 0.01)[0]]

las_res2 = []
for i in las_res:
    sz = sz_dict[i[0]]
    prnt_szx = sz * i[1]
    las_res2.append((i) + (sz, prnt_szx,))

las_res_gnrs = [i[0] for i in las_res2]
las_res3 = [i[3] for i in las_res2]

nps(las_res_gnrs, las_res3, las_res3)
nps(las_res_gnrs, [i[1] for i in las_res], 1)
res_df = pd.DataFrame(las_res2, columns = ['gnr', 'coef', 'sz', 'sz2'])


# *** plotting klds, maybe they also have scree-plot like stuff


valid_ids = np.where(ar_cb[gnr_ind[gnr]] < math.inf)[0]
xs = [gnrs[i] for i in valid_ids]
ys = [ar_cb[gnr_ind[gnr],i] for i in valid_ids]


res2_df = pd.DataFrame(xs, columns = ['genre'])
res2_df['kld'] = ys

v1 = res2_df.sort_values('kld')['kld']
v2 = res2_df.sort_values('kld')['genre']

nps(v2, v1, 1)
# hm might actually work, there's a turn right at the start

s1 = res2_df.sort_values('kld')['kld']
s1_diff1 = s1[0:50].diff().diff()

nps(range(50),s1_diff1, 1)

s1_diff1 = s1_diff1.reset_index()
npl(s1_diff1['kld'])
# probably needs some smoothing, then find max of second differencing? 



# ** artist genres
# see if there's a clear distinction in terms of features


artsts_l = [i.lower() for i in artsts]

artst_gnrs = []
for gnr in gnrs_l:
    if gnr in artsts_l:
        artst_gnrs.append(gnr)

# i think it's just easier to filter on percentage of unique artists


# this opens the can of worms of integrating artist information as well FUCK ME
