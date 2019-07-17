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

def asym_sim(gx, gnrs, vdx):
    """generates asym mat based on weights scaled by max, making more equally distributed genres more general"""
    cmps = all_cmps_crubgs(gnrs, vdx, 'product')

    all_sims = vertex_similarity(gx, 'dice', vertex_pairs = cmps)
    
    # don't think there's much sense in weights: i have weights because i standardize in np.hist
    # if i don't all proportionality constraints are of and larger ones will swallow small ones
    # but if i use weights the outdegree becomes the same which makes asymmetric similarity pointless
    # -> asymmetric similarity through overlap requires variation in out_degree
    # all_sims = vertex_similarity(gx, 'dice', vertex_pairs = cmps, eweight = w_std2)

    sims_rows = np.split(all_sims, len(gnrs))
    sims_ar = np.array(sims_rows)

    # deg_vec = [gx.vertex(vdx[i]).out_degree(weight=w_std2) for i in gnrs]
    deg_vec = [gx.vertex(vdx[i]).out_degree() for i in gnrs]

    # equivalent to adding the two outdegrees together 
    deg_ar = np.array([deg_vec]*len(gnrs))
    deg_ar2 = (deg_ar + np.array([deg_vec]*len(gnrs)).transpose())/2

    # see how much is actually in common, equivalent of multiplication with similarity
    cmn_ar = deg_ar2*sims_ar

    # see the percentage of what is in common for each genre
    ovlp_ar = cmn_ar/deg_ar
    return(ovlp_ar)


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
