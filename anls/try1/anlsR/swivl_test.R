## * libs
q
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


## * stuff from elsewhere
if (!require(pacman)) install.packages("pacman")

orca <- read.table("http://www.stats4life.se/data/oralca.txt")

orca_mut <- orca %>%
    mutate(
        text = paste("Subject ID = ", id, "<br>", "Time = ", time, "<br>", "Event = ",  
                     event, "<br>", "Age = ", round(age, 2), "<br>", "Stage = ", stage)
    )

ggplot(orca_mut, aes(x = id, y = time, text = text)) +
    geom_linerange(aes(ymin = 0, ymax = time)) +
    geom_point(aes(shape = event, color = event), stroke = 1, cex = 2) +
    scale_shape_manual(values = c(1, 3, 4)) +
    labs(y = "Time (years)", x = "Subject ID", tooltip = "text")+
    coord_flip() + theme_classic() 



grid.arrange(
  ggplot(orca, aes(x = id, y = time)) +
  geom_linerange(aes(ymin = 0, ymax = time)) +
  geom_point(aes(shape = event, color = event), stroke = 1, cex = 2) +
  scale_shape_manual(values = c(1, 3, 4)) + guides(shape = F, color = F) +
  labs(y = "Time (years)", x = "Subject ID") + coord_flip() + theme_classic(),
  orca %>%
  mutate(age_orig = age,
         age_end = age + time) %>%
  ggplot(aes(x = id, y = age_end)) +
  geom_linerange(aes(ymin = age_orig, ymax = age_end)) +
  geom_point(aes(shape = event, color = event), stroke = 1, cex = 2) +
  scale_shape_manual(values = c(1, 3, 4)) + guides(fill = FALSE) +
  labs(y = "Age (years)", x = "Subject ID") + coord_flip() + theme_classic(),
  ncol = 2
)

su_obj <- Surv(orca$time, orca$all)
str(su_obj)


## * merge data together

res_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/12_weeks_26k_full_min20/'
res_files = list.files(res_dir)

dfc <- read.csv(paste0(res_dir, res_files[1]))

for (i in res_files[2:length(res_files)]){
    res_fl = paste0(res_dir, i)
    dfx <- read.csv(res_fl)
    dfc <- rbind(dfc, dfx)
}

dfc[which(is.na(dfc$acst_cor_mean)),]$acst_cor_mean <- 1


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
dfc2$inftns <- dfc2$prnt3_dvrg_mean
dfc2$inftns_sqrd <- dfc2$inftns^2

## ** distinctiveness

dfc2$disctns <- dfc2$cohrt_mean_cos_dists_wtd_mean
dfc2$cohrt_vol <- dfc2$cohrt_vol_sum_mean

## what's the reasoning behind using min/max?
## vast majority of genres (93%) that die out just have 1 ptn -> min/max makes no difference
## is more for p-hacking the sample i guess



## cohrt_vars = c("cohrt_mean_non_inf_mean","cohrt_mean_cos_dists_wtd_mean","cohrt_mean_cos_dists_uwtd_mean","cohrt_len_mean", "cohrt_vol_sum_mean", "cohrt_vol_mean_mean", "cohrt_vol_sd_mean")

## chart.Correlation(dfc2[,cohrt_vars])

## main things are cohort distance and cohort volume,
## cohort_len is same as prnt_sz (prnt_odg_mean)


## ALSO USE THE penl_chirns here

## *** potential parents

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
dfc2$prnt_sz <- dfc2$prnt_odg_mean


## ** agreement

ag_vars_set  <- c('acst_cor_mean', 'nbr_ptns', 'prnt_sims', 'chirn_sims')
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

sd_set1 <- c("prnt3_dvrg_sd","mean_prnt_dvrg_sd","prnt_odg_sd","cohrt_pct_inf_sd","gnr_gini_sd")
## "unq_artsts_sd" doesn't fit so well
sd_set2 <- c("avg_age_sd", "age_sd_sd", "nbr_rlss_tprd_sd", "ttl_size_sd", "prop_rls_size_sd", "volm_sd")

sd_pca <- principal(dfc2[,c(sd_set1, sd_set2)], nfactors=3)

## does taking the mean make measures invalid?
chart.Correlation(dfc2[,c(sd_set1, sd_set2)])

## chart.Correlation(dfc2[,var_set2])

## pcax <- principal(dfc2[,var_set2])

## ** size
var_set_sz <- c('unq_artsts_mean', 'volm_mean', 'sz_raw_mean', 'nbr_rlss_tprd_mean')
## chart.Correlation(dfc2[,var_set_sz])

## sz_cmpst <- principal(dfc2[,var_set_sz])
## hist(sz_cmpst$scores, breaks=100)
## sz_rscld <- scales::rescale(sz_cmpst$scores, to=c(0.1, 10))
## sz_cmpst_log <- log(sz_rscld)
## hist(sz_cmpst_log, breaks=100)

dfc2$sz <- log(dfc2$volm_mean)
dfc2$new_rlss <- log(dfc2$nbr_rlss_tprd_mean+1)

## no PCA, just use logs of vol and new_rlss as indicators

## ** controls
ctrl_vars = c('avg_weight_rel_mean','spngns_std_mean', 'dist_mean_mean', 'gnr_gini_mean', 'avg_age_mean', 'cohrt_vol',  'len_penl_prnts_100_mean', 'sz', 'new_rlss', 'gnr_age')
## chart.Correlation(dfc2[,var_set_crols])

## all seems sufficiently uncorrelated

## ** rel variables

dfc2$tp_id2 <- dfc2$tp_id+1

## inf_vars <- c('inftns',  'disctns')
inf_vars <- c('inftns', 'inftns_sqrd', 'disctns') 

agr_vars <- c('acst_cor_mean', 'nbr_ptns', 'prnt_sims')

leg_vars <- c('prnt_sz')
## leg_vars2 <- c('prnt_sz', 'I(prnt_sz^2)')

## is there even a separate legitimation thing ?

all_vars <- c(inf_vars, agr_vars, leg_vars, ctrl_vars)


## standardizing
not_scale <- c('nbr_ptns', 'gnr_age')

dfc3 <- as.data.frame(scale(dfc2[,all_vars[all_vars %!in% not_scale]]))
dfc3 <- cbind(dfc3, dfc2[,c('X', 'tp_id', 'tp_id2', 'event', not_scale)])

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
fit_ctrl <- phreg(f_ctrl, data=dfc3, cuts = seq(10,15),dist = 'pch')

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

agr_ctrl_vars <- paste(c(agr_vars, ctrl_vars), collapse = ' + ')
f_agr_ctrl <- as.formula(paste(c(dv, agr_ctrl_vars), collapse = ' ~ '))
fit_agr_ctrl <- phreg(f_agr_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')

res_agr_ctrl <- phreg_mdlr(fit_agr_ctrl)
screenreg(list(res_ctrl, res_inf_ctrl, res_agr_ctrl))

## ** legitimation
leg_ctrl_vars <- paste(c(leg_vars, ctrl_vars), collapse = ' + ')
f_leg_ctrl <- as.formula(paste(c(dv, leg_ctrl_vars), collapse = ' ~ '))
fit_leg_ctrl <- phreg(f_leg_ctrl, data=dfc3, cuts = seq(1,28),dist = 'pch')
res_leg_ctrl <- phreg_mdlr(fit_leg_ctrl)

screenreg(list(res_ctrl, res_inf_ctrl, res_agr_ctrl, res_leg_ctrl))
## ** all
all_vars_cbnd <- paste(c(all_vars), collapse = ' + ')
f_all_vars <- as.formula(paste(c(dv, all_vars_cbnd), collapse = ' ~ '))
fit_all_vars <- phreg(f_all_vars, data=dfc3, cuts = seq(1,28),dist = 'pch')
res_all_vars <- phreg_mdlr(fit_all_vars)


screenreg(list(res_ctrl, res_inf_ctrl, res_agr_ctrl, res_leg_ctrl, res_all_vars))


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


fit3_breaks <- pchreg(f, data=dfc2, breaks = 35)
fit3_no_breaks <- pchreg(f, data=dfc2[which(dfc2$tp_id > 0),])

plot(fit3_breaks)


## * coxph


f_ctrl_coxph <- as.formula(paste(c(dv, paste(c(ctrl_vars_cbnd, 'cluster(X)'), collapse = '+')), collapse = ' ~ '))

res_coxph_ctrl <- coxph(f_ctrl_coxph, data=dfc3)
## ## res_coxreg <- coxreg(f_ctrl_coxreg, data=dfc3)
## screenreg(res_coxph)

## coxreg takes forever -> not good



f_all_coxph <- as.formula(paste(c(dv, paste(c(all_vars_cbnd, 'cluster(X)'), collapse = ' + ')), collapse = ' ~' ))
res_coxph_all <- coxph(f_all_coxph, data = dfc3)
screenreg(list(res_coxph_ctrl,res_coxph_all))


       

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




## * testing
## ** documentation of pchreg
n <- 1000
x <- runif(n)
time <- rnorm(n, 1 + x, 1 + x)
cens <- rnorm(n,2,2)
y <- pmin(time,cens) # censored variable
d <- (time <= cens) # indicator of the event

model <- pchreg(Surv(y,d) ~ x, breaks = 20)


## ** bromstrom
require(eha)
data(fert)
f12 <- fert[fert$parity == 1, ]
f12$Y <- Surv(f12$next.ivl, f12$event)
head(f12)


fit.pch <- phreg(Surv(next.ivl, event) ~ age + year + ses, data = f12, dist = "pch", cuts = c(4, 8, 12))
fit.c <- coxreg(Surv(next.ivl, event) ~ age + year + ses,data = f12)
fit.pch2 <- phreg(Surv(next.ivl, event) ~ age + year + ses, data = f12, dist = "pch", cuts = 1:13)

fit.pch
check.dist(fit.c, fit.pch2)
## apparently should compare between parametric (phreg) and semi-parametric coxreg models?

plot(fit.pch2, fn = "haz")

plot(fit.pch)
plot(fit.pch2)


check.dist(fit2, fit2)
## search for
## - repeated measures
## - time-varying covariates




## ??
## newcgd <- tmerge(data1=cgd0[, 1:13], data2=cgd0, id=id, tstop=futime)
## coxph(formula = Surv(tstart, tstop, infect) ~ treat + inherit + steroids +cluster(id), data = newcgd)




## dfc3 <- dfc2[-which(dfc2$prnt_odg_wtd==Inf),]
## dfc3 <- dfc2[-0,]

## dfc4 <- dfc2[-10,]

fit1 <- coxph(Surv(tp_id, tp_id2, event) ~ inftns + disctns + cluster(X), data=dfc2)
fit1 <- coxph







fit <- coxph(Surv(tp_id, tp_id2, event) ~ avg_weight_rel_mean + cohrt_pct_inf_mean + sz_raw_mean + prnt3_dvrg_mean + prnt_odg_mean + prnt_odg_wtd_mean + spngns_mean + spngns_std_mean + cluster(X), data= dfc2)
## size is fairly obvious
## prnt3_dvrg is more interesting tbh: more informative -> more likely to die

fit2 <- coxph(Surv(tp_id, tp_id2, event) ~ avg_weight_rel_mean + cohrt_pct_inf_mean + 
                 sz_raw_mean + prnt3_dvrg_mean + prnt_odg_mean + prnt_odg_wtd_mean +
                 spngns_mean + spngns_std_mean + cluster(X), data= dfc2)
## including cohrt_mean_non_inf and deleting a bunch -> cohrt_pct_inf suddenly significant, prnt3 dvrg not anymore


fit3 <- coxph(Surv(tp_id, tp_id2, event) ~ cohrt_pct_inf_mean + prnt3_dvrg_mean + I(prnt3_dvrg_mean^2)+
                 sz_raw_mean +  prnt_odg_mean + prnt_odg_wtd_mean +
                 spngns_mean + spngns_std_mean + cluster(X), data= dfc3)


fit_cheat <- coxph(Surv(tp_id, tp_id2, event) ~ avg_weight_rel + cohrt_pct_inf + sz_raw + prnt3_dvrg + prnt_odg + prnt_odg_wtd + spngns + spngns_std, data= dfc3)




## not clusterin -> average weight relative becomes suddenly significant ahahha

screenreg(fit)

## there seem to be be issues with prnt3_dvrg and prnt_odt_wtd, both have infinites but shouldn't
## is one weird case for both


## i'm losing like half of my cases for DVs to cheesiness
## is there a solution to cheesiness?

## only count last block?

## don't like it: the underlying process is not a clear-cut death-or-alive state
## maybe really fitness more generally does better job of capturing dynamics


## add age/tenure

## make general function

## doesn't seem so bad? most (1511/1755) seem coherent 
## might be due to later start: most are present at the end, but don't have 15 entries because they are younger
## -> not as holy as worst case
## might be that holy ones are disproportionally in failed ones tho

## ** just treat it as unbalanced panel?

formla <- "(1 | X)"
formla2 <- "sz_raw ~ (1 | X ) + prnt3_dvrg"

fit_fe <- lmer(formla2, data=dfc3)

## ** use plm? claims to be good for unbalanced panels
## ahhh so much stuff and no dieuwke

dfc1.5 <- dfc[-which(dfc$prnt3_dvrg == Inf),]

fit_plm_fe <- plm(sz_raw ~ prnt3_dvrg + cohrt_pct_inf + spngns, data=dfc1.5, model = 'within', na.action = 'na.exclude', index = 'X')

fit_plm_fe_lag <- plm(sz_raw ~ lag(prnt3_dvrg,1) + lag(cohrt_pct_inf,1) + lag(spngns,1), data=dfc1.5, model = 'within', na.action = 'na.exclude', index = 'X')

fit_plm_re_lag <- plm(sz_raw ~ lag(prnt3_dvrg,1) + lag(cohrt_pct_inf,1) + lag(spngns,1), data=dfc1.5, model = 'random', na.action = 'na.exclude', index = c('X'))

fit_plm_re2 <- plm(sz_raw ~ prnt3_dvrg + cohrt_pct_inf + spngns, data=dfc3, model = 'random', na.action = 'na.exclude', index = c('X', 'tp_id'))

screenreg(list(fit_plm_fe_lag, fit_plm_re, fit_plm_re_lag))
phtest(fit_plm_fe_lag, fit_plm_re_lag)

plmtest(fit_plm_re_lag, type='bp', effect = 'time')


## ** DV transformation

## idk just do log?
## piazzai: doesn't seem to use log?
## but uses poisson

## ** pglm
library(pglm)
fit_g_fe <- pglm(log(sz_raw) ~ lag(prnt3_dvrg,1) + lag(cohrt_pct_inf,1) + lag(spngns,1),
                 data = dfc1.5[which(dfc1.5$tp_id > 7),],
                 family = 'poisson',
                 model = 'within',
                 index = 'tp_id')


## ** time series descripties
tsx <- aggregate(dfc1.5$sz_raw, list(dfc1.5$tp_id), sum)
barplot(tsx$x)


## ** other stuff

## need example of data structure: long/short?
## how are multiple observations treated?

## x <- n.0.disease.1year<-data.frame(event=rep(0,times=177), right=rep(1,times=177))
## x2 <- n.0.disease.2year<-data.frame(event=rep(0,times=937), right=rep(2,times=937))

## how to get multiple time points
## or do i need to summarize to time to death?
## but then i lose information on covariates
## Negro Winemaking doesn't sound like they summarized into one time to death variables


data(cancer)
fit <- survfit(Surv(time, status) ~ sex, data = cancer)
print(fit)

ggsurvplot(fit,
          pval = TRUE, conf.int = TRUE,
          risk.table = TRUE, # Add risk table
          risk.table.col = "strata", # Change risk table color by groups
          linetype = "strata", # Change line type by groups
          surv.median.line = "hv", # Specify median survival
          ggtheme = theme_bw(), # Change ggplot2 theme
          palette = c("#E7B800", "#2E9FDF"))

newcgd <- tmerge(data1=cgd0[, 1:13], data2=cgd0, id=id, tstop=futime)
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime1))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime2))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime3))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime4))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime5))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime6))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime7))
newcgd <- tmerge(newcgd, newcgd, id, enum=cumtdc(tstart))

attr(newcgd, "tcount")

coxph(formula = Surv(tstart, tstop, infect) ~ treat + inherit + steroids +cluster(id), data = newcgd)

## this looks nice, basically what i want
## multiple observations for each case
## clustering by genre

## tstart and tstop are not the same across cases
## wonder what they should be based on?

## is because futime (follow up time) is not completely uniform i guess? 
## infect gets added by decoding the etimes i guess

## not really time dependent covariates but that comes later?

## * transplant as time dependent variable

jasa$subject <- 1:nrow(jasa) #we need an identifier variable
tdata <- with(jasa, data.frame(subject = subject,
                               futime= pmax(.5, fu.date - accept.dt),
                               txtime= ifelse(tx.date== fu.date,
                               (tx.date -accept.dt) -.5,
                               (tx.date - accept.dt)),
                               fustat = fustat
                               ))
sdata <- tmerge(jasa, tdata, id=subject,
                death = event(futime, fustat),
                trt= tdc(txtime),
                options= list(idname="subject"))
attr(sdata, "tcount")

sdata$age <- sdata$age -48

sdata$year <- as.numeric(sdata$accept.dt - as.Date("1967-10-01"))/365.25

coxph(Surv(tstart, tstop, death) ~ age*trt + surgery + year + cluster(subject),  data= sdata, ties="breslow")
coxph(Surv(tstart, tstop, death) ~ age*trt + surgery + year,  data= sdata, ties="breslow")

## have to see what this formula does exactly
## again varying times

## *** something else

library(texreg)


data(mort)
fit <- phreg(Surv(enter, exit, event) ~ ses, data = mort)
fit
plot(fit)
fit.cr <- coxreg(Surv(enter, exit, event) ~ ses, data = mort)
check.dist(fit.cr, fit)

## *** pch
library(pch)

## no examples??

## *** https://stats.stackexchange.com/questions/236382/how-to-fit-piece-wise-exponential-model-in-r


library(devtools)
devtools::install_github("adibender/pammtools")
library(pammtools)
library(survival)
library(ggplot2)
theme_set(theme_bw())
library(mgcv)

# load example data
data(tumor)

## transform to piece-wise exponential data (PED)
ped_tumor <- as_ped(Surv(days, status) ~ ., data = tumor, cut = seq(0, 3000, by = 100))

# Fit the Piece-wise-exponential Additive Model (PAM) instead of PEM
# with piece-wise constant hazards, term s(tend), and constant covariate effects:
pam <- mgcv::gam(ped_status ~ s(tend) + age + sex + complications, 
  data = ped_tumor, family = poisson(), offset = offset)
summary(pam)
## plot baseline hazard: 
ped_tumor %>% 
  make_newdata(tend=unique(tend), age = c(0), sex = c("male"), complications=c("no")) %>% 
  add_hazard(pam) %>% 
  ggplot(aes(x = tstart, y = hazard)) + 
  geom_stepribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2) +
  geom_step()


## *** Bornstrom book
data(oldmort)

om <- oldmort[oldmort$enter == 60, ]
om <- age.window(om, c(60, 70))
om$m.id <- om$f.id <- om$imr.birth <- om$birthplace <- NULL
om$birthdate <- om$ses.50 <- NULL
om1 <- survSplit(om, cut = 61:69, start = "enter", end = "exit", event = "event", episode = "agegrp")
om1$agegrp <- factor(om1$agegrp, labels = 60:69)
om1 <- om1[order(om1$id, om1$enter), ]
head(om1)

rownames(om1) <- 1:NROW(om1)
om1$id <- as.numeric(as.factor(om1$id))
head(om1)

recs <- tapply(om1$id, om1$id, length)
barplot(table(recs))

om1$exit <- om1$enter <- NULL

om2 <- reshape(om1, v.names = c("event", "civ", "region"),
               idvar = "id", direction = "wide",
               timevar = "agegrp")
names(om2)

om3 <- reshape(om2, direction = 'long', idvar = 'id', varying = 3:32)
om3 <- om3[order(om3$id, om3$time), ]
om3 <- om3[!is.na(om3$event), ]

om3$time <- as.factor(om3$time)

om3[1:11, ]

fit.glm <- glm(event ~ sex + civ + region + time, family = binomial(link = cloglog), data = om3)

drop1(fit.glm, test='Chisq')

library(glmmML)
fit.boot <- glmmML::glmmboot(event ~ sex + civ + region, cluster = time, family = binomial(link = cloglog), data = om3)


om3$exit <- as.numeric(as.character(om3$time))
om3$enter <- om3$exit - 0.5
fit.ML <- coxreg(Surv(enter, exit, event) ~ sex + civ + region, method = "ml", data = om3)
fit.ML

plot(fit.ML, fn = 'cum', xlim = c(60,70))
plot(fit.ML, fn = 'surv', xlim = c(60,70))

fit2.glm <- glm(event ~ (sex + civ + region) * time, family = binomial(link = cloglog), data = om3)
drop1(fit2.glm, test = "Chisq")

x <- drop1(fit2.glm, test = "Chisq")

## *** trying to skip reshaping, works just fine -> reshaping not needed
om1$agegrp <- as.numeric(as.character(om1$agegrp))
om1$agegrp2 <- as.numeric(as.character(om1$agegrp)) - 1
fit.ML2 <- coxreg(Surv(agegrp2, agegrp, event) ~ sex + civ + region, method = "ml", data = om1)




## *** timedp.pdf
## seems to have clustering by id

fit.x2 <- coxph(Surv(agegrp2, agegrp, event) ~ sex + civ + region + +cluster(id), data = om1)
## seems to produce same estimates as coxreg

fit.x3 <- coxph(Surv(agegrp2, agegrp, event) ~ sex + civ + region, data = om1)
## also exactly same stuff as when not clustering
## makes me wonder if cluster(id) is doing what's it's supposed to
## coefs are same, but SEs/p-values are slightly different, seem to be up to 10% bigger when accounting for clustering
## or sometimes also smaller?

## i think that only works if some condition is fulfilled
## last dda task?
7





## * stuff


