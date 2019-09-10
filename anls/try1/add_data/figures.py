## * example distributions

rc('figure', figsize=(11.69,8.27))

def sb_pltr(ax, ttl, row, nbr, xs, ys, xlbl, ylbl, gnr):
    # plt.subplot(ttl, row, nbr)
    ax.plot(xs, ys, '.-', label = gnr)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)

gnrs2 = ['rap', 'metal', 'ambient', 'rock', 'pop', 'punk']

nbr_bins = 5

vrbls2 = vrbls[0:5]
vrbl_lbls = {'dncblt': 'danceability', 'gender': 'gender', 'timb_brt':'timbre', 'tonal':'tonality', 'voice':'vocality'}


ax_lbl = [str(round(i,2)) + '-' +str(round(i+1/nbr_bins,2)) for i in np.arange(0,1,1/nbr_bins)]

fig, axs = plt.subplots(5)
st = fig.suptitle('Genre Probability Distributions')



for gnr in gnrs2: 
    c = 0
    dfcx = acst_gnr_dict[gnr]
    dfcx['wt'] = dfcx['cnt']*dfcx['rel_weight']

    for v in vrbls2:

        ax = axs[c]

        # wrap into genre function
        
        a1, a0 = np.histogram(dfcx[v], bins=nbr_bins, weights=dfcx['wt'])
        a2 = [i/sum(a1) for i in a1]
        v2 = vrbl_lbls[v]
        ax.set_ylabel(v2)

        ttl = 5
        c +=1
        sb_pltr(ax ,ttl, 1, c, ax_lbl, a2, 'sub-dimension', v2, gnr)

plt.legend(ncol=3, bbox_to_anchor = (1, -0.4))


fig.set_size_inches(5, 8)
fig.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.9)

fig.savefig('/home/johannes/Dropbox/gsss/thesis/text/figures/ills.pdf', dpi = 200)

plt.show()
plt.close()


## ** time scale

client = Client(host='localhost', password='anudora', database='frrl')

times= client.execute('select time_d, count(time_d) from logs group by time_d')

time_ar = np.array(times)
time_ar2 = time_ar[time_ar[:,0] < datetime.date(datetime(2014, 9, 19, 0, 0))]

fig = plt.figure()
fig.set_size_inches(7, 3)
plt.plot(time_ar2[:,0], time_ar2[:,1])
fig.suptitle('MLHD Daily Listening Events')

fig.savefig('/home/johannes/Dropbox/gsss/thesis/text/figures/time.pdf', dpi = 200)

plt.show()


# * KLD vis



import numpy as np
from numpy import random

sigma = 10

sigmas = [round(i,4) for i in np.arange(0.3, 4.025, 0.025)]

nbr_cls = 33

x_vlus = bins = np.array(np.linspace(-4, 4, nbr_cls))[:, np.newaxis]



entrs = []
for sigma in sigmas: 
    ys = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
    y_std = ys/sum(ys)
    entr = entropy(y_std)
    entrs.append(entr[0])
    print(sigma, entr)

npl(entrs)

all_res = []

for s1 in sigmas:
    d1 = 1/(s1 * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * s1**2))
    d1_std = d1/sum(d1)
    
    res_vec = []
    
    for s2 in sigmas:
        d2 = 1/(s2 * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * s2**2))
        d2_std = d2/sum(d2)
        
        kld = entropy(d2_std, d1_std)
        res_vec.append(kld[0])
    all_res.append(res_vec)
    
# it is more unlikely to observe the general d1_std (sigma 4) from a specific d2 (sigma 0.3) (kld 17) than it is to observe the specific d2 as a result from the general d1 (kld 1.57)???
# i am so confuse
# i think it's the tails: it's harder for d2 to lift its tails (close to 0) to generate d1 than for d1 to flatten itself? 

# it is more difficult for a true low entropy distribution to generate a high entropy dist than it is for a high entropy dist to generate a low entropy dist? 

# -> parents (high entropy) can easier generate dists of children (low entropy) than children can dists of parents? 

res_ar = np.array(all_res)
import matplotlib.colors as colors

plt.rc('figure', figsize=(6.5,5))

res_ar2 = res_ar + 0.001


fig = plt.figure()
ax = fig.add_subplot(111)

pcm = ax.pcolor(res_ar2, 
                norm = colors.LogNorm(vmin=res_ar2.min(), vmax = res_ar2.max()),
                cmap = 'viridis', 
                rasterized = True)

fig.colorbar(pcm)


tick_pos = [0, 28, 68, 108, 148]
# tick_pos_y = [len(sigmas) -i -1 for i in [0, 28, 68, 108, 148]]

ax.xaxis.set_ticks_position('top')
ax.set_xticks(tick_pos)
ax.set_yticks(tick_pos)

# tick_pos_y.reverse()

ax.set_xticklabels([sigmas[i] for i in tick_pos], fontsize = 10)
ax.set_yticklabels([sigmas[i] for i in tick_pos], fontsize = 10)

plt.xlabel(r'$\sigma_P$', fontsize = 18)
ylbl = plt.ylabel(r'$\sigma_Q$', fontsize = 18,labelpad= 15)
ylbl.set_rotation(0)

plt.title('KLDs (log-normalized) between normal distributions', fontsize = 16, x =0.6)

ax.invert_yaxis()
fig.savefig('/home/johannes/Dropbox/gsss/thesis/text/figures/kld.pdf', dpi = 150)

plt.show()



# pcm = ax[0].pcolor(X, Y, Z1,
#                    norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()),
#                    cmap='PuBu_r')


# ax.set_xticklabels(['']+alpha)
# ax.set_yticklabels(['']+alpha)


# plt.show()

# goes row-wise

# concept entropy as predictor? 

# isn't there a direction issue? a general distribution (parent) is more likely to be generated by a more specific one (child) than that general generates child ??

# oh snap
# it means similarity goes in wrong direction? 
# hard rock is less dissimilar (more similar) from rock than rock is from 

# test with metal and funk metal
g1 = 'metal'
g2 = 'funk metal'

ar_cb[gnr_ind[g1], gnr_ind[g2]] # approximating metal with funk metal costs 0.054
ar_cb[gnr_ind[g2], gnr_ind[g1]] # approximating funk metal with metal costs 0.0504

entropy(acst_mat[gnr_ind[g1]], acst_mat[gnr_ind[g2]])
entropy(acst_mat[gnr_ind[g2]], acst_mat[gnr_ind[g1]])

# hmm that looks good so far: variant is more similar to prototype than other way around
# need general framework tho

# do i even have right measure? 
# i mean i checked that so many times.. 
# g_kld2 seems to be using correct one (0.0504 for metal -> funk metal)
# yup: ar_cb[g1,g2] is how well g1 is approximated with g2


kld3_el = []

c = 0
for i in kld2_el:
    g1, g2 = i[0], i[1]
    rev_kld = ar_cb[gnr_ind[g1], gnr_ind[g2]]
    
    item = list(i)
    item.append(rev_kld)

    kld3_el.append(item)

    c+=1
    
from scipy.stats import sem

from scipy.stats import ttest_ind



kld3_df = pd.DataFrame(kld3_el, columns = ['g1', 'g2', 'g1_g2', 'g2_g1'])
kld3_df['dif'] = kld3_df['g1_g2'] - kld3_df['g2_g1']
nph(kld3_df['dif'])
np.mean(kld3_df['dif'])
sem(kld3_df['dif'])


ttest_ind(kld3_df['g1_g2'], kld3_df['g2_g1'])
# points somewhat in the expected direction (children are closer to parents than parents to children)
# t-test is significant

# but still theoretical tension: should be more expensive to approximate specific ones wiht general ones
# does it hinge on the assumption that parents are bigger? 
# bigger: more entropy -> more evenly spread


x

# asymmetry as predictor? 




# entropy has only weak correlation with size? 0.12 with original, 0.18 with log transformed volume
# np.corrcoef(np.log(df_res['volm']), df_res['entrp'])
# np.corrcoef(df_res['volm'], df_res['entrp'])
