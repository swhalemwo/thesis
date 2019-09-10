## * libs

library(ggplot2)
library(PerformanceAnalytics)
library(texreg)
library(psych)
library(texreg)

'%!in%' <- function(x,y)!('%in%'(x,y))
len = length
pacman::p_load(tidyverse, survival, ggfortify, survminer, plotly, gridExtra, 
               Epi, KMsurv, gnm, cmprsk, mstate, flexsurv, splines, epitools, 
               eha, shiny, ctqr, scales)

## extraction functions

phreg_mdlr <- function(mdl, imprvmnt){
    vrbls <- mdl$covars
    coefs <- mdl$coefficients
    vars <- diag(mdl$var)
    ses <- sqrt(vars)
    pvlus <- 2*pnorm(abs(coefs/ses), lower.tail = FALSE)
    
    gof.vlus <- c(mdl$events, mdl$ttr, max(mdl$loglik))
    ## maybe add  model improvement? could do manually later
    ## anova doesn't like 
                  
    gof.names <- c('events', 'genre-timeperiods', 'max. log. likelihood')
    

    res <- createTexreg(coef.names = vrbls,
                 coef = coefs,
                 se = ses,
                 pvalues = pvlus,
                 gof.names = gof.names,
                 gof = gof.vlus
                 )
    return(res)
}

## logdiff = 2*(-1075.196 + 1239.430)
## pchisq(logdiff, df=2, lower.tail=FALSE)


## * merge data together

res_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/no_ptns_xxx/'
res_files = list.files(res_dir)

dfc <- read.csv(paste0(res_dir, res_files[1]))

for (i in res_files[2:length(res_files)]){
    res_fl = paste0(res_dir, i)
    dfx <- read.csv(res_fl)
    dfc <- rbind(dfc, dfx)
}

dfc[which(is.na(dfc$acst_cor_mean)),]$acst_cor_mean <- 1

## dfc <- dfc[which(dfc$sz_raw > 50),]

## just rbind everything together
## check consistency with length of range: more range covered than periods -> inconsistent
## for those that are consistent: if max range < max(range for unit) -> failure at that time
## not sure if that lags properly but whatever

## df_cut <- dfc[-which(dfc$sz_raw_mean < 30 & dfc$sz_raw_sd == 0),]
## dfc <- df_cut

## cut run ptn_proc with max and min to be able to filter better
## that's p-hacking
## can re-use partition matrices tho

dfc$X <- as.factor(dfc$X)
dfc <- dfc[order(dfc$X, dfc$tp_id),]

dfc$event <- 0

gnr_inf <- aggregate(dfc$tp_id, list(dfc$X), min)
gnr_inf2 <- aggregate(dfc$tp_id, list(dfc$X), max)
## 364 failed
## that's not nothing but not nearly as much as i wish

ded_gnrs <- gnr_inf2[which(gnr_inf2$x < max(gnr_inf2$x)-1),]$Group.1

gnr_inf3 <- aggregate(dfc$tp_id, list(dfc$X), length)

## see which are complete

df_gnrs <- cbind(gnr_inf, gnr_inf2$x, gnr_inf3$x)
names(df_gnrs) <- c('X', 'min', 'max', 'nb_entrs')


df_gnrs$max2 <- df_gnrs$max +1
df_gnrs$chk <- df_gnrs$max2 - df_gnrs$nb_entrs

## cheese genres due to holes like swiss cheese
cheese_gnrs <- df_gnrs[which(df_gnrs$min != df_gnrs$chk),'X']
usbl_cheese_gnrs <- df_gnrs[df_gnrs$X %in% cheese_gnrs & df_gnrs$max <= max(df_gnrs$max)-3 ,]$X


length(which(ded_gnrs %!in% cheese_gnrs))
## i mean still have 233 complete ded gnrs

## dfc2 <- dfc[which(dfc$X %!in% cheese_gnrs),]

dfc2 <- merge(dfc, df_gnrs, by = 'X')

dfc2$gnr_age <- dfc2$tp_id - dfc2$min

ded_gnrs2 <- unique(dfc2$X)[which(unique(dfc2$X) %in% ded_gnrs)]
ded_gnrs2 <- ded_gnrs[which(ded_gnrs %!in% cheese_gnrs)]

ded_gnrs3 <- paste(c(as.character(ded_gnrs2), as.character(usbl_cheese_gnrs)))

## set dying out for ded genres
for (i in ded_gnrs3){

    dfx <- dfc2[which(dfc2$X ==i),]
    max_tp <- max(dfx$tp)

    dfc2[which(dfc2$X == i & dfc2$tp_id == max_tp),]$event <- 1
    ## print(c(i, dim(dfx)))
}

deds <- aggregate(dfc2$event, list(dfc2$tp_id), sum)

## barplot(deds[,2])
## ** more filtering: make min_tag_apprnc a bit higher? 
## min_szs <- aggregate(dfc2$sz_raw_mean, list(dfc2$X), min)

## min_szs <- aggregate(dfc2[which(dfc2$X %in% ded_gnrs2),]$sz_raw_mean, list(dfc2[which(dfc2$X %in% ded_gnrs2),]$X), min)

## dfc2[which(dfc2$X in ded_gnrs2)]$sz_raw_mean

## don't think it would lead to more dead gnrs:
## might at most decrease the percentage of ded genres among cheese gnrs if cheese gnrs are more among under 30/35/X than dead genres
##  but that doesn't turn cheese genres into real ded gnrs, but rather removes whole lot of dead genres

## but ACKUALLY it might turn alive gnrs ded
## not super clean: gnr can have sz_raw_min under new threshold but still be alive if shared by multiple groups
## just remove those with sz_raw_sd = 0



## * variable construction
## ** informativeness
## dfc2$inftns <- dfc2$prnt3_dvrg_mean
## dfc2$inftns_sqrd <- dfc2$inftns^2

## need to rename if i don't do partitions

## *** no partitioning
dfc2$inftns <- dfc2$prnt3_dvrg
dfc2$inftns_sqrd <- dfc2$inftns^2

## ** distinctiveness

## dfc2$disctns <- dfc2$cohrt_mean_cos_dists_wtd_mean
## dfc2$cohrt_vol <- dfc2$cohrt_vol_sum_mean

## *** just one partition

dfc2$disctns <- dfc2$cohrt_mean_cos_dists_wtd

dfc2$cohrt_dom <- log(dfc2$volm/dfc2$cohrt_vol_sum)


## cohrt_rel_sz <- dfc2$volm/dfc2$cohrt_vol_mean
## doesn't seem so nice,
## can't control for genre size

cohrt_vol_mean <- dfc2$cohrt_vol_sum/dfc2$cohrt_len

dfc2$cohrt_rel_sz <- log(dfc2$volm/cohrt_vol_mean)




## what's the reasoning behind using min/max?
## vast majority of genres (93%) that die out just have 1 ptn -> min/max makes no difference
## is more for p-hacking the sample i guess



## cohrt_vars = c("cohrt_mean_non_inf_mean","cohrt_mean_cos_dists_wtd_mean","cohrt_mean_cos_dists_uwtd_mean","cohrt_len_mean", "cohrt_vol_sum_mean", "cohrt_vol_mean_mean", "cohrt_vol_sd_mean")

## chart.Correlation(dfc2[,cohrt_vars])

## main things are cohort distance and cohort volume,
## cohort_len is same as prnt_sz (prnt_odg_mean)


## ALSO USE THE penl_chirns here

## *** Potential parents

len_vars = c("len_penl_prnts_100_mean", "len_penl_prnts_0.1_mean", "len_penl_prnts_0.2_mean", "len_penl_prnts_0.3_mean")

len_vars_addgs = c("len_penl_prnts_100_mean", "len_penl_prnts_0.1_mean", "len_penl_prnts_0.2_mean", "len_penl_prnts_0.3_mean", 'inftns', 'disctns')

## chart.Correlation(dfc2[,len_vars_addgs])

# len 0.1-0.3 highly correlated (0.7-.9)
## moderate/high correlations between inftns/distcns and the len variables
## -> more potential parents <-> less informativeness/distinctive; makes kinda sense tbh: closer KLDs means there are more of them in the different cutoff rates

mean_vars_addgs = c("mean_penl_prnts_100_mean", "mean_penl_prnts_0.1_mean", "mean_penl_prnts_0.2_mean", "mean_penl_prnts_0.3_mean", 'inftns', 'disctns')
## chart.Correlation(dfc2[,mean_vars_addgs])
## values are weird
## negative/small positive correlations between 100 and 0.1, 0.2, 0.3

sum_vars_addgs = c("sum_penl_prnts_100_mean", "sum_penl_prnts_0.1_mean", "sum_penl_prnts_0.2_mean", "sum_penl_prnts_0.3_mean", 'inftns', 'disctns')

## chart.Correlation(dfc2[,sum_vars_addgs])
## similar patterns as with means
## tbh i can't see the meaning of sums: adding a bunch of divergences together doesn't make something more/less distinct
## can also be many small/few big ones -> stick with lens adn means so far

penl_parnt_vars_addgs = c("len_penl_prnts_100_mean", "len_penl_prnts_0.1_mean", "len_penl_prnts_0.2_mean", "len_penl_prnts_0.3_mean", "mean_penl_prnts_100_mean", "mean_penl_prnts_0.1_mean", "mean_penl_prnts_0.2_mean", "mean_penl_prnts_0.3_mean", 'inftns', 'disctns')
## chart.Correlation(dfc2[,penl_parnt_vars_addgs])
## stronger correlations within lens than sums for 0.1-0.3
## len_penl_prnts_100 doesn't correlate highly with anything,
## mean_penl_prnts_100 with informmativeness (0.8) and the lens 0.1-0.3 (-0.65-0.75)

## i need a theory of what they mean
## len_pnl_prnts_100 is basically a measure of how isolated a genre is overall
## but so is mean_pnl_prnts
## it's quite abstract tho.. maybe it should be weighted by size, but that would make it even more complex

## maybe len_pnl_prnts is fine due to simplicty
## also good for not being correlated with inftns/distcns, although it kinda is supposed to measure distcns? 
## different theoretical justification? distcns is just to cohornt, len_pnl_prnts_100 is to overall?
## the 100 parts are maybe also preferable over the other cutoffs due to no large amount of zeroes, i.e. nicer dists

## the fact that mean_penl_prnts_100 is highly correlated with inftns (prnt3_dvrg_mean) is quite telling
## it means you can predict how similar it is to all by just taking (the mean of) the 3 closest

## anyways just use len_penl_prnts_100 for now




## ** parent size

## *** no partitions
dfc2$prnt_sz <- log(dfc2$prnt_plcnt)
dfc2$prnt_sz_sqrd <- dfc2$prnt_sz^2
dfc2$prnt_sz_sd <- log(dfc2$prnt_plcnt_sd)

## ** agreement

## ag_vars_set  <- c('acst_cor_mean', 'nbr_ptns', 'prnt_sims', 'chirn_sims')


## chart.Correlation(dfc2[,ag_vars_set])

## nbr_prnts unrelated to acst_cor_mean, prtn_sims, chirn_sims
## they are sufficient on their own tho
## tbh don't like prtn_sims and chrin_sims because they are effectively conditional on having at least two partitions having the genre
## but so is acst_cor: if just 1 partition, correlation is 1
## also not so nice distribtutions
## tbh negative correlations not that bad, just should be high
## more partitions lead to slightly less acst_correlations huh
## well more comparisons -> more space for disagreement


## agr_w_nbr_ptns <- principal(dfc2[,c('nbr_ptns','acst_cor_mean', 'prnt_sims', 'chirn_sims')])
## agr_wo_nbr_ptns <- principal(dfc2[,c('acst_cor_mean', 'prnt_sims', 'chirn_sims')])
## nbr_prnts  decreases fit, but would probably still pass

## dfc2$agr_w_nbr_ptns <- agr_w_nbr_ptns$scores
## dfc2$agr_wo_nbr_ptns <- agr_wo_nbr_ptns$scores
## quite similar tbh, 97 cor


## don't think i should use PCA: demands too much, also don't control for clustering
## just use acst_cor_mean, nbr_ptns, prnt_similarity on their own

## *** sds

## hmm not exactly clear what to make of it, so many variables
## maybe break into related groups? could mirror the other topics
## agreement might vary depending on the topic

## sd_set1 <- c("prnt3_dvrg_sd","mean_prnt_dvrg_sd","prnt_odg_sd","cohrt_pct_inf_sd","gnr_gini_sd")
## ## "unq_artsts_sd" doesn't fit so well
## sd_set2 <- c("avg_age_sd", "age_sd_sd", "nbr_rlss_tprd_sd", "ttl_size_sd", "prop_rls_size_sd", "volm_sd")

## sd_pca <- principal(dfc2[,c(sd_set1, sd_set2)], nfactors=3)

## ## does taking the mean make measures invalid?
## chart.Correlation(dfc2[,c(sd_set1, sd_set2)])

## chart.Correlation(dfc2[,var_set2])

## pcax <- principal(dfc2[,var_set2])

## ** density

## *** no partition


dfc2$dens_vol <- log(dfc2$cohrt_vol_sum + dfc2$volm)
dfc2$dens_len <- dfc2$cohrt_len

dfc2$dens_vol_sqrd <- dfc2$dens_vol^2
dfc2$dens_len_sqrd <- dfc2$dens_len^2

## worth to distinguish between len and vol because might be that amount of genres or playcounts matters
## also no no high correlation

## ** size
## var_set_sz <- c('unq_artsts_mean', 'volm_mean', 'sz_raw_mean', 'nbr_rlss_tprd_mean')
## chart.Correlation(dfc2[,var_set_sz])

## sz_cmpst <- principal(dfc2[,var_set_sz])
## hist(sz_cmpst$scores, breaks=100)
## sz_rscld <- scales::rescale(sz_cmpst$scores, to=c(0.1, 10))
## sz_cmpst_log <- log(sz_rscld)
## hist(sz_cmpst_log, breaks=100)

## dfc2$sz <- log(dfc2$volm_mean)
## dfc2$new_rlss <- log(dfc2$nbr_rlss_tprd_mean+1)

## *** no partitions
dfc2$sz <- log(dfc2$volm)
dfc2$new_rlss <- log(dfc2$nbr_rlss_tprd+1)


## no PCA, just use logs of vol and new_rlss as indicators

## ** controls
## ctrl_vars = c('avg_weight_rel_mean','spngns_std_mean', 'dist_mean_mean', 'gnr_gini_mean', 'avg_age_mean', 'cohrt_vol',  'len_penl_prnts_100_mean', 'sz', 'new_rlss', 'gnr_age')

## *** one partition

ctrl_vars <- c('avg_weight_rel_wtd', 'cos_sims_mean_wtd', 'gnr_gini', 'avg_age', 'sz', 'new_rlss', 'gnr_age')
## 'len_penl_prnts_100'
## 'spngns_std'

## chart.Correlation(dfc2[,var_set_crols])
## all seems sufficiently uncorrelated

## ** rel variables

dfc2$tp_id2 <- dfc2$tp_id+1

## inf_vars <- c('inftns',  'disctns')

## inf_vars <- c('inftns', 'inftns_sqrd', 'disctns') 

## agr_vars <- c('acst_cor_mean', 'nbr_ptns', 'prnt_sims')

## leg_vars <- c('prnt_sz')
## leg_vars2 <- c('prnt_sz', 'I(prnt_sz^2)')

## is there even a separate legitimation thing ?

## all_vars <- c(inf_vars, agr_vars, leg_vars, ctrl_vars)

## *** no partitions

inf_vars <- c('inftns', 'inftns_sqrd', 'disctns')
dens_vars <- c('dens_vol', 'dens_len', 'dens_vol_sqrd', 'dens_len_sqrd', 'cohrt_dom', 'cohrt_rel_sz')
prnt_vars <- c('prnt_sz', 'prnt_sz_sqrd')

all_vars <- c(inf_vars, dens_vars,  prnt_vars, ctrl_vars)

## standardizing
## not_scale <- c('nbr_ptns', 'gnr_age')
not_scale <- c('gnr_age')

## scale(dfc2[,all_vars[all_vars %!in% not_scale]], center = FALSE, scale = apply(dfc2[,all_vars[all_vars %!in% not_scale]], 2, sd, na.rm = T))

dfc3 <- dfc2[0,all_vars]

for (i in unique(dfc2$tp_id)){
    dfc3_prep <- as.data.frame(scale(dfc2[which(dfc2$tp_id == i),all_vars[all_vars %!in% not_scale]], center=FALSE,
                                     scale = apply(dfc2[which(dfc2$tp_id == i),all_vars[all_vars %!in% not_scale]],
                                                   2, sd, na.rm = T)))
                               
    dfc3_prep2 <- cbind(dfc3_prep, dfc2[which(dfc2$tp_id == i),c('X', 'tp_id', 'tp_id2', 'event', not_scale)])
    dfc3 <- rbind(dfc3, dfc3_prep2)
}
 
       
## dfc3 <- as.data.frame(scale(dfc2[,all_vars[all_vars %!in% not_scale]], center=FALSE, scale = apply(dfc2[,all_vars[all_vars %!in% not_scale]], 2, sd, na.rm = T)))

## dfc3 <- cbind(dfc3, dfc2[,c('X', 'tp_id', 'tp_id2', 'event', not_scale)])

## could add log size
## size negatively related to distinctiveness and informativeness: kinda like that stuff that's closer together is bigger; more distant things don't grow so well
## not clear if distinctivenss on its own makes so much sense without a strict hierarchy

# chart.Correlation(dfc2[,all_vars])

## * phreg

dv <- 'Surv(tp_id, tp_id2, event)'

## dv2 <- 'Surv(tp_id, event)' is used for stuff without time-variation i think
## fit1 <- coxph(f, data=dfc2)
## fit1_reg <- coxreg(f, data=dfc2)

## ** controls
ctrl_vars_cbnd <- paste(ctrl_vars, collapse = ' + ')
f_ctrl <- as.formula(paste(c(dv, ctrl_vars_cbnd), collapse = ' ~ '))
fit_ctrl <- phreg(f_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')

res_ctrl <- phreg_mdlr(fit_ctrl, None)
screenreg(list(res_ctrl))

## ** informativeness
inf_ctrl_vars <- paste(c(inf_vars, ctrl_vars), collapse = ' + ')
f_inf_ctrl <- as.formula(paste(c(dv, inf_ctrl_vars), collapse = ' ~ '))
fit_inf_ctrl <- phreg(f_inf_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')

res_inf_ctrl <- phreg_mdlr(fit_inf_ctrl)

screenreg(list(res_ctrl, res_inf_ctrl))
# res_inf

## ** agreement

## agr_ctrl_vars <- paste(c(agr_vars, ctrl_vars), collapse = ' + ')
## f_agr_ctrl <- as.formula(paste(c(dv, agr_ctrl_vars), collapse = ' ~ '))
## fit_agr_ctrl <- phreg(f_agr_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')

## res_agr_ctrl <- phreg_mdlr(fit_agr_ctrl)
## screenreg(list(res_ctrl, res_inf_ctrl, res_agr_ctrl))

## ** legitimation
## leg_ctrl_vars <- paste(c(leg_vars, ctrl_vars), collapse = ' + ')
## f_leg_ctrl <- as.formula(paste(c(dv, leg_ctrl_vars), collapse = ' ~ '))
## fit_leg_ctrl <- phreg(f_leg_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')
## res_leg_ctrl <- phreg_mdlr(fit_leg_ctrl)

## screenreg(list(res_ctrl, res_inf_ctrl, res_agr_ctrl, res_leg_ctrl))

## ** density
dens_ctrl_vars <- paste(c(dens_vars, ctrl_vars), collapse = ' + ')
f_dens_ctrl <- as.formula(paste(c(dv, dens_ctrl_vars), collapse = ' ~ '))
fit_dens_ctrl <- phreg(f_dens_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')
res_dens_ctrl <- phreg_mdlr(fit_dens_ctrl)

## ** prnt vars
prnt_ctrl_vars <- paste(c(prnt_vars, ctrl_vars), collapse = ' + ')
f_prnt_ctrl <- as.formula(paste(c(dv, prnt_ctrl_vars),  collapse = ' ~  '))
fit_prnt_ctrl <- phreg(f_prnt_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')
res_prnt_ctrl <- phreg_mdlr(fit_prnt_ctrl)


## ** all
all_vars_cbnd <- paste(c(all_vars), collapse = ' + ')
f_all_vars <- as.formula(paste(c(dv, all_vars_cbnd), collapse = ' ~ '))
fit_all_vars <- phreg(f_all_vars, data=dfc3, cuts = seq(1,28),dist = 'pch')
res_all_vars <- phreg_mdlr(fit_all_vars)


## screenreg(list(res_ctrl, res_inf_ctrl, res_agr_ctrl, res_leg_ctrl, res_all_vars))
screenreg(list(res_ctrl, res_inf_ctrl, res_dens_ctrl, res_prnt_ctrl, res_all_vars))
## dens_vol positive, dens_len negative
## squred ones reversed
## does it make sense to do it without size controls? 


## strongest impact of agreement
## more partitions: less likely to disappear
## but more agreement -> more likely to die out? check if in correct direction
## higher informativeness: more likely to die out -> add quadratic term
## no influence of distinctiveness, but is shitty measurement atm
## prnt_sze: small effect

## is there straightforward link between informativeness and density? don't think so

## summary(fit2_cuts)
## plot(fit2_cuts)
## plot(fit2_cuts, 'haz')

## check.dist(fit1_reg, fit2_cuts)
## ttl <- aggregate(dfc2$X, list(dfc2$tp_id), length)

## ratio_man <- deds$x/ttl$x
## plot(seq(0,28),ratio_man, type='s')
## cum_man <- cumsum(ratio_man)
## plot(cum_man, type='l')

## x <- as.data.frame(cbind(t(t(ratio_man)), t(fit2_cuts$hazards)))



## hmm manual hazard ratio and that by function produces different results
## same shapes, but manual one is 40% higher
## i think it's due to lag in manual?
## nope, changing type of plotting doesn't change it overall
## in partitcular the 4 last columns: manual last4 to last 2 are somewhat similar, while in fit the third last is much higher
## whatever m8
## adding I(inftns^2) changes hazard function, makes it more similar to manual wrt to last 4 spells

## plot(fit2_no_cuts)
# not specifying cuts makes plots look weird
## fit2_frcd <- as(fit2, 'phreg', strict=TRUE)

## wonder if i should correct times to proper duration
## can also probably just ADD AGE adn be fine

## pchreg produces ugly result, stick with phreg
##


## fit3_breaks <- pchreg(f, data=dfc2, breaks = 35)
## fit3_no_breaks <- pchreg(f, data=dfc2[which(dfc2$tp_id > 0),])

## plot(fit3_breaks)




## * predicted probability
## need them to see u-shape effect effect of informativeness
## just group event by informativeness?

seqx <- seq(-2, 3)

dfc3$inftns_disc <- round(dfc3$inftns)
dfc3$inftns_disc2 <- round(dfc3$inftns,1)

mfx_cheap <- as.data.frame.matrix(table(dfc3$inftns_disc, dfc3$event))

mfx_cheap$ratio <- mfx_cheap[,'1']/mfx_cheap[,'0']


## ** custom AAP
## /*takes each individual in the data in turn, treats them as they if were a man 
## (regardless of the actual gender of the individual) leaving the values of all other 
## independent variables as observed; computes probability of this individual having a
## job; then repeats but this time treating individual as if they were a woman.
## Takes the average of the predictions for women and men*/

## -> for all values in range, give them to cases, mean/sd for each 

colnames = fit_inf_ctrl$covars
dfc3_ap <- dfc3[,colnames]

squared = FALSE
vrbl <- 'sz'
if (squared == TRUE){
    vrbl_sqrd <- paste0(vrbl, '_sqrd')
    sqr_sclr <- mean(dfc3_ap[,vrbl_sqrd]/dfc3[,vrbl]^2)
}

coef_vec <- fit_inf_ctrl$coefficients
coef_mat <- t(replicate(nrow(dfc3), coef_vec))

mean_vec <- colMeans(dfc3_ap)
colnames(coef_mat) <- colnames
names(coef_vec) <- colnames

coef_ses <- sqrt(diag(fit_inf_ctrl$var))
se_rel <- coef_ses[names(coef_ses) == vrbl]

## coef_mat_lo <- coef_mat_hi <- coef_mat
## coef_mat_lo[,vrbl] <- coef_mat_lo[,vrbl]-se_rel
## coef_mat_hi[,vrbl] <- coef_mat_hi[,vrbl]+se_rel

coef_vec_hi <- coef_vec_lo <- coef_vec
coef_vec_hi[vrbl]  <- coef_vec_hi[vrbl]+ 1.96*se_rel
coef_vec_lo[vrbl]  <- coef_vec_lo[vrbl] - 1.96*se_rel

if (squared == TRUE){
    se_rel_sqrd <- coef_ses[names(coef_ses) == vrbl_sqrd]
    coef_mat_lo[,vrbl_sqrd] <- coef_mat_lo[,vrbl_sqrd]-se_rel_sqrd
    coef_mat_hi[,vrbl_sqrd] <- coef_mat_hi[,vrbl_sqrd]+se_rel_sqrd
}

# can't just standardize again, need way to get squared variables to proper values
## base_rate <- mean(rowSums(dfc3_ap * coef_mat))
## base_rate_lo <- mean(rowSums(dfc3_ap * coef_mat_lo))
## base_rate_hi <- mean(rowSums(dfc3_ap * coef_mat_hi))
base_rate <- sum(mean_vec*coef_vec)
base_rate_lo <- sum(mean_vec*coef_vec_lo)
base_rate_hi <- sum(mean_vec*coef_vec_hi)

vrbl_vlus <- seq(from=min(dfc3_ap[,vrbl]), to=max(dfc3_ap[,vrbl]), length.out = 15)


res <- c()
res_lo <- c()
res_hi <- c()


for (i in vrbl_vlus){
    dfc3_ap[,vrbl] <- i

    if (squared == TRUE){
        dfc3_ap[,vrbl_sqrd] <- (i^2)*sqr_sclr
    }

    predx <- dfc3_ap * coef_mat
    predx2 <- rowSums(predx)
    resids <- dfc2$event - exp(predx2)
    se_x <- sqrt(t(as.matrix(predx)) %*% fit_inf_ctrl$var %*% as.data.frame(predx))

    
    ## pred_hi <- rowSums(dfc3_ap*coef_mat_hi)
    ## pred_lo <- rowSums(dfc3_ap*coef_mat_lo)

    ## res <- c(res, mean(predx2))
    ## res_hi <- c(res_hi, mean(pred_hi))
    ## res_lo <- c(res_lo, mean(pred_lo))

    mean_vec[vrbl] <- i
    resx <- sum(mean_vec*coef_vec)

    res_lox <- sum(mean_vec*coef_vec_lo)
    res_hix <- sum(mean_vec*coef_vec_hi)

    res <- c(res, resx)
    res_hi <- c(res_hi, res_hix)
    res_lo <- c(res_lo, resx_lox)
    
}

## resx is log of hazards function: f(t)/S(t):
## risk of dying while having survived so far
## exp(resx) is actual hazard function: 0.69 chance of dying for smallest genres
## still seems like you should divide by baserate, like value for mean (0.0267 chance)
## -> small genres are 26.4x more likely to die

## what happens at 1?
## you get there when exp(res) and exp(base_rate) are same, i.e. at mean
## i think it's because mean is the comparison
## and the hi/lo things are also estimated with the mean


## stata does for each individual
## with error thing
## but would take up 25gb ram
## noo way




res_df <- as.data.frame(cbind(exp(res)/exp(base_rate),
                              exp(res_lo)/exp(base_rate_lo),
                              exp(res_hi)/exp(base_rate_hi),
                              vrbl_vlus))



## res_df_inftns <- res_df

names(res_df) <- c('res', 'lo', 'hi', 'vlus')

res_df$hi2 <- apply(res_df, 1, function(x){max(x['hi'],x['lo'])})
res_df$lo2 <- apply(res_df, 1, function(x){min(x['hi'],x['lo'])})

res_df$vrbl <- vrbl
res_df_c <- rbind(res_df_inftns, res_df)

res_df_c$vrbl2 <- res_df_c$vrbl

res_df_c$xs <- rep(seq(1:15),2)

## for some reason difference becomes 0 at 1
## , group = vrbl2

ggplot(res_df_inftns, aes(x=xs, y = res)) + 
    geom_line()+
    geom_ribbon(aes(ymin=lo2, ymax=hi2), alpha=0.25) + 
    ## ylim(0, 12)+
    theme_bw()


df2 <- data.frame(supp=rep(c("VC", "OJ"), each=3),
                dose=rep(c("D0.5", "D1", "D2"),2),
                len=c(6.8, 15, 33, 4.2, 10, 29.5))

head(df2)

## idk if it makes much sense without CIs.. would be kinda pointless to not have CIs to also visually show why some things don't matter

ggplot(data=df2, aes(x=dose, y=len, group=supp)) +
  geom_line()

## ** avg_age
plot(vrbl_vlus,unlist(lapply(vrbl_vlus, function(x){exp(-0.062*x)})), type = 'l')
lines(vrbl_vlus,unlist(lapply(vrbl_vlus, function(x){exp((-0.062-0.049)*x)})), type = 'l')
lines(vrbl_vlus,unlist(lapply(vrbl_vlus, function(x){exp((-0.062+0.049)*x)})), type = 'l')

## ** size

plot(vrbl_vlus,unlist(lapply(vrbl_vlus, function(x){exp(-1.818*x)})), type = 'l')
lines(vrbl_vlus,unlist(lapply(vrbl_vlus, function(x){exp((-1.818-0.152)*x)})), type = 'l')
lines(vrbl_vlus,unlist(lapply(vrbl_vlus, function(x){exp((-1.818+0.152)*x)})), type = 'l')

## ** inftns

plot(vrbl_vlus, unlist(lapply(vrbl_vlus, function(x){exp(0.563*x) * exp((-0.255)*sqr_sclr*x^2)})), type='l')
plot(vrbl_vlus, unlist(lapply(vrbl_vlus, function(x){exp((0.563+0.222)*x) * exp((-0.255+0.157)*sqr_sclr*x^2)})), type='l')
lines(vrbl_vlus, unlist(lapply(vrbl_vlus, function(x){exp((0.563-0.222)*x) * exp((-0.255-0.157)*sqr_sclr*x^2)})), type='l')

plot(unlist(lapply(seq(30), function(x){exp((0.563+0.222)*x) * exp((-0.255+0.157)*sqr_sclr*x^2)})), type='l')




## bootstrapping?
## sounds quite expensive tbh
## also unclear what's the point is: i have good estimates for my variables, 
    
## stuff to consider
## - probability to die given that having survived
## - variation in baserate -> should use time_specific baserates
## predicted baserate different from actual baserate
## probably due to age?

## could just use curves without confidence intervals?
## 


plot(exp(1)^res/rep(exp(1)^base_rate, 10), type = 'l')
plot(exp(1)^res_lo/rep(exp(1)^base_rate, 10), type = 'l')
plot(exp(1)^res_hi/rep(exp(1)^base_rate, 10), type = 'l')



## plot(exp(1)^res, type = 'l')
lines(exp(1)^(res-1.96*ses), type = 'l')
lines(exp(1)^(res+1.96*ses), type = 'l')


se <- function(x) sqrt(var(x)/length(x))


## is probability? NOPE
## coefs are logs of risk ratios
## exponentiated coefs are relative risks/hazard ratios
## so what does hazard ratio of 0.3 mean? A hazard ratio of 0.333 tells you that the hazard rate in the treatment group is one third of that in the control group.
## let's just do it like Piazzai and call them multipliers


    
## * coxph

ctrl_vars_cbnd2 <- paste(ctrl_vars[1:2], collapse = ' + ')
f_ctrl_coxph <- as.formula(paste(c(dv, paste(c(ctrl_vars_cbnd2), collapse = '+')), collapse = ' ~ '))

res_coxph_ctrl <- coxph(f_ctrl_coxph, data=dfc3, ties='breslow')
summary(res_coxph_ctrl)
screenreg(res_coxph_ctrl)

preds <- predict(res_coxph_ctrl, type = 'expected')


hist(exp(1)^-preds, breaks = 100)

pred_df <- as.data.frame(preds)
## pred_df$X <- rownames(pred_df)

pred_df2 <- cbind(pred_df, dfc3$inftns_disc2)
names(pred_df2) <- c('pred', 'inftns_disc2')

mfx2 <- aggregate(pred_df2$pred, list(pred_df2$inftns_disc2), mean)

x <- margins(res_coxph_ctrl, type = 'risk')
# margins doesn't seem to like whatever type


y <- stdReg(res_coxph_ctrl)



hist(preds)

dfc3$pre

res_coxreg <- coxreg(f_ctrl_coxreg, data=dfc3)

## screenreg(res_coxph)

## coxreg takes forever -> not good



f_all_coxph <- as.formula(paste(c(dv, paste(c(all_vars_cbnd, 'cluster(X)'), collapse = ' + ')), collapse = ' ~' ))
res_coxph_all <- coxph(f_all_coxph, data = dfc3)
screenreg(list(res_coxph_ctrl,res_coxph_all))




n <- 1000
Z <- rnorm(n)
X <- rnorm(n, mean=Z)
T <- rexp(n, rate=exp(X+Z+X*Z)) #survival time
C <- rexp(n, rate=exp(X+Z+X*Z)) #censoring time
U <- pmin(T, C) #time at risk
D <- as.numeric(T < C) #event indicator
dd <- data.frame(Z, X, U, D)
fit <- coxph(formula=Surv(U, D)~X+Z+X*Z, data=dd, method="breslow")
fit.std <- stdCoxph(fit=fit, data=dd, X="X", x=seq(-1,1,0.5), t=1:5)
print(summary(fit.std, t=3))
plot(fit.std)


stdCoxph(res_coxph_ctrl, data = dfc3, X="spngns_std", x = c(-4, -2, 0, 2), t=5:6)

specials <- pmatch(c("strata(","cluster(","tt("), attr(terms(fit$formula), "variables"))

       

## * plm

library(plm)


## ** controls
ctrl_vars_plm <- ctrl_vars[ctrl_vars %!in% c('sz', 'new_rlss')]
dv_plm <- 'sz'

ctrl_vars_plm_cbnd <- paste(ctrl_vars_plm, collapse = ' + ')
f_plm_ctrl <- as.formula(paste(c(dv_plm, ctrl_vars_plm_cbnd), collapse = ' ~ '))
fit_ctrl_plm <- plm(f_plm_ctrl, data = dfc3, effect = 'twoways', model = 'within', index = c('X', 'tp_id'))
summary(fit_ctrl_plm)
screenreg(fit_ctrl_plm)

## FE: time-invariant ommitted variables
## well there's on statistical trick against omitted variables


## ** all 

dfc3$len_penl_prnts <- dfc3$len_penl_prnts_100_mean

all_vars_cbnd <- paste(c(all_vars[all_vars %!in% c('sz', 'new_rlss')]), collapse = ' + ')
all_vars_cbnd <- "inftns + inftns_sqrd + disctns + acst_cor_mean + nbr_ptns + prnt_sims + prnt_sz + avg_weight_rel_mean + spngns_std_mean + dist_mean_mean + gnr_gini_mean + avg_age_mean + cohrt_vol + len_penl_prnts"

f_all_plm <- as.formula(paste(c(dv_plm, all_vars_cbnd), collapse = ' ~ '))
fit_ctrl_plm <- plm(f_all_plm, data = dfc3, effect = 'twoways', model = 'within', index = c('X', 'tp_id'))
summary(fit_ctrl_plm)
screenreg(fit_ctrl_plm)

# GNR AGE
## x <- fixef(fit_ctrl_plm,effect="")


## fit_ctrl_pglm <- pglm(f_plm_ctrl, data = dfc3, effect = 'twoways', model = 'pooling', index = c('X', 'tp_id'), family = poisson, R = 1000)
## fit_ctrl_pglm
## screenreg(fit_ctrl_pglm)



## * mets

write.csv(dfc3, file = "dfc3.csv")
dfc3 <- read.csv("dfc3.csv")

library(mets)

fitx <- gof(fit_ctrl)

f_ctrl <- as.formula(paste(c(dv, ctrl_vars_cbnd), collapse = ' ~ '))
fit_ctrl <- mets::phreg(f_ctrl, data=dfc3)
summary(fit_ctrl)
, cuts = seq(1,28),dist = 'pch')

re

basehazplot.phreg(fit_ctrl)
gofM.phreg(f_ctrl, )

## * timereg
library(timereg)

dfc3$genre <- dfc3$X

fitx <- cox.aalen(Surv(tp_id, tp_id2,event) ~ prop(sz) + prop(dist_mean), data=dfc3, id = dfc3$genre, cluster = dfc3$genre, max.timepoint.sim = 10, basesim=1, propodds = 1)
summary(fitx)

## Z=as.matrix(dfc3[1:10,c('sz', 'dist_mean')])
Z = cbind(c(2,4,6), rep(mean(dfc3$dist_mean), 3))

pred <- predict(fitx, Z=Z, n.sim=0)
plot(pred,multiple=1,se=2,uniform=5,col=1:3,lty=1:10)

## hmm not sure if useful
## i mean i could predict stuff for each variable and process that further
## but that doesn't make it any easier



summary(fitx)
plot(fitx)

data(sTRACE)
head(sTRACE)


out<-cox.aalen(Surv(time,status==9)~prop(age)+prop(sex)+
prop(diabetes)+chf+vf,
data=sTRACE,max.time=7,n.sim=0,resample.iid=1)

pout<-predict(out,X=rbind(c(1,0,0),c(1,1,0)),Z=rbind(c(55,0,1),c(60,1,1)))
head(pout$S0[,1:5]); head(pout$se.S0[,1:5])
par(mfrow=c(2,2))
plot(pout,multiple=1,se=0,uniform=0,col=1:2,lty=1:2)
plot(pout,multiple=0,se=1,uniform=2,col=1:2)


           

par(mfrow=c(1,2))
ss <- cox.aalen(Surv(time,status==9)~+prop(vf),data=sTRACE,robust=0)
par(mfrow=c(1,2))
plot(ss)


## hmm i can just predict as they do: not use the entire df, just change the value of interest, keep rest at mean
