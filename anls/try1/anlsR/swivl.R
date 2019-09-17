## * libs

library(ggplot2)
library(PerformanceAnalytics)
library(texreg)
library(psych)
library(xtable)
library(fBasics)
library(readr)
library(Hmisc)
library(collections)

'%!in%' <- function(x,y)!('%in%'(x,y))
len = length
pacman::p_load(tidyverse, survival, ggfortify, survminer, plotly, gridExtra, 
               Epi, KMsurv, gnm, cmprsk, mstate, flexsurv, splines, epitools, 
               eha, shiny, ctqr, scales)

texreg_cleaner <- function(infile){
    tex <- read_file(infile)
    tex1 <- gsub("\\$\\^", "", tex)
    tex2 <- gsub("\\^\\{", "", tex1)
    tex3 <- gsub("\\}\\$", "", tex2)
    tex4 <- gsub("\\& \\$", "\\& ", tex3)
    tex5 <- gsub("\\)\\$", "\\)", tex4)
    tex6 <- gsub("\\$", "", tex5)

    write_file(tex6, infile)
}


## extraction functions

phreg_mdlr <- function(mdl, d, imprvmnt){
    vrbls <- mdl$covars
    vrbl_names = unlist(lapply(vrbls, function(x){d$get(x)}))
    coefs <- mdl$coefficients
    vars <- diag(mdl$var)
    ses <- sqrt(vars)
    pvlus <- 2*pnorm(abs(coefs/ses), lower.tail = FALSE)
    
    gof.vlus <- c(mdl$events, mdl$ttr, max(mdl$loglik), mdl$df)
    ## maybe add  model improvement? could do manually later
    ## anova doesn't like 
                  
    gof.names <- c('events', 'genre-timeperiods', 'Log-likelihood', 'df')
    

    res <- createTexreg(coef.names = vrbl_names,
                 coef = coefs,
                 se = ses,
                 pvalues = pvlus,
                 gof.names = gof.names,
                 gof = gof.vlus
                 )
    return(res)
}

## * merge data together

res_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/harsh_narrow/'
res_files = list.files(res_dir)

dfc <- read.csv(paste0(res_dir, res_files[1]))

for (i in res_files[2:length(res_files)]){
    res_fl = paste0(res_dir, i)
    dfx <- read.csv(res_fl)
    dfc <- rbind(dfc, dfx)
}



## dfc_bu <- dfc
dfc <- dfc_bu

## filter if needed
## dfc <- dfc[which(dfc$nbr_rlss_tprd > 0),]

## * find ded genres


## INST_MIN = 20 # instantiation threshold: genre not alive before reaching this point
## NBR_ENTRS = 3 # minimum number of entries
## NOT USED BECAUSE CREATING UNDED GNRS
EMPT_END <- 4 # number of period at end in which genre may not appear
## if 9 weeks: 4
## if 6 weeks: 5
## since i'm not using NBR_ENTRS, rather be cautious and use 5 for 9 weeks as well

# am I making sure that NBR_ENTRS has occurred /before/ EMPT_END? 


dfc$X <- as.factor(dfc$X)
dfc <- dfc[order(dfc$X, dfc$tp_id),]

## first_equal_to2 = function(x, value) {
##     result = logical(length(x))
##     result[match(value, x)] = TRUE
##     result
## }

## dfc$thd_bin <- 0
## dfc[dfc$sz_raw >= INST_MIN,]$thd_bin <- TRUE

## dfc$first_thd <- 0

## ## find first time point where genre appears (> INST_MIN)
## dx <- dfc %>% group_by(X) %>% mutate(first_thd = first_equal_to2(thd_bin, 1))

## first_thds <- dx[dx$first_thd,c('X', 'tp_id')]
## names(first_thds) <- c('X', 'first_tp')

## dx2 <- merge(dx, first_thds, by = 'X')
## dx2$imtur <- 0
## dx2[which(dx2$tp_id < dx2$first_tp),]$imtur <- 1

## DELETE IMMATURE? before instantiation
## NEEDS TO BE PUT INTO PYTHON
## seems to be working: no more imturs

## dx3 <- dx2[-which(dx2$imtur ==1),]
## dfc <- dx3
## dfc <- dx2

dfc$event <- 0

gnr_inf <- aggregate(dfc$tp_id, list(dfc$X), min)
gnr_inf2 <- aggregate(dfc$tp_id, list(dfc$X), max)
## 364 failed
## that's not nothing but not nearly as much as i wish

ded_gnrs <- gnr_inf2[which(gnr_inf2$x <= max(gnr_inf2$x)-EMPT_END),]$Group.1

gnr_inf3 <- aggregate(dfc$tp_id, list(dfc$X), length)

gnr_max_sz <- aggregate(sz_raw ~ X, dfc, max)


df_gnrs <- cbind(gnr_inf, gnr_inf2$x, gnr_inf3$x, gnr_max_sz$sz_raw)
names(df_gnrs) <- c('X', 'min', 'max', 'nb_entrs', 'max_sz')

## are event accumulated before timeframe?
## ded_gnrs only has those which died before 3 before end
## yup: 

df_gnrs$max2 <- df_gnrs$max +1
df_gnrs$chk <- df_gnrs$max2 - df_gnrs$nb_entrs


dfc2 <- merge(dfc, df_gnrs, by = 'X')
dfc2 <- dfc2[order(dfc2$X),]

dfc2$gnr_age <- dfc2$tp_id - dfc2$min
## dfc2$gnr_age2 <- dfc2$tp_id - dfc2$min
## i want to know which entry something is (der wievielte)
## 
ages2 <- unlist(lapply(df_gnrs$nb_entrs, function(x){seq(1:x)}))
dfc2$gnr_age2 <- ages2

## delete late time period?

## ded_gnrs2 <- unique(dfc2$X)[which(unique(dfc2$X) %in% ded_gnrs)]
## ded_gnrs2 <- ded_gnrs[which(ded_gnrs %!in% cheese_gnrs)]

## ded_gnrs3 <- paste(c(as.character(ded_gnrs), as.character(usbl_cheese_gnrs)))

## i think i should not use NBR_ENTRS: unded gnrs are even deader than dead gnrs
## also no way to find unded gnrs in python because they depend on nbr entries
## aliveness of a genre at point in time should not depend on status in other times

## ded_gnrs4 <- df_gnrs[df_gnrs$X %in% ded_gnrs & df_gnrs$nb_entrs >= NBR_ENTRS,]$X
## ded_gnrs2 <- ded_gnrs

## dfx <- dfc2[dfc2$X %in% ded_gnrs2,]
## max_szs <- aggregate(dfx$sz_raw, list(dfx$X), max)

## ded_gnrs4 <- max_szs[max_szs$x > 20,]$Group.1
## len(ded_gnrs4)

dfc2$max_tp <- FALSE
dfc2$max_tp[which(dfc2$tp_id == dfc2$max)] <- TRUE

dfc2$event[dfc2$X %in% ded_gnrs & dfc2$max_tp ==TRUE] <- 1

##  ** set dying out for ded genres

## dfc2_bu <- dfc2
## dfc2 <- dfc2_bu

## delete undead genres
## unded_gnrs <- ded_gnrs[ded_gnrs %!in% ded_gnrs4]
## len(unded_gnrs)

## if (len(unded_gnrs) > 0){
##     print('adsf')
##     dfc2 <- dfc2[-which(dfc2$X %in% unded_gnrs),]
## }

dfc2 <- dfc2[dfc2$tp_id <= max(dfc2$tp_id)-EMPT_END,]

## delete cell_cmbs missing values
## atm removes only two events, which seems ok i guess

## dfc22 <- dfc2[-which(is.na(dfc2$cell_cmbs_prnt3_dvrg)),]
## dfc2 <- dfc22

table(dfc2$event)
## dfc2_bu <- dfc2
## dfc2 <- dfc2_bu

deds <- aggregate(dfc2$event, list(dfc2$tp_id), sum)
barplot(deds$x)

## also need to check how much it goes up afterwards
## or i can just skip that

## i can also combine the multiple reasons how genres have died

## ** renaming into desired hierarchical set

smpl_ftrs_names <- names(dfc)[unlist(lapply(names(dfc), function(x){grepl('smpl_ftrs', x)}))]

for (i in smpl_ftrs_names){
    
    new_vrbl_name <- substr(i, 11, 100)

    dfc2[,new_vrbl_name] <- dfc2[,i]
}




## * variable construction
## ** informativeness

dfc2$inftns <- log(dfc2$prnt3_dvrg)
dfc2$inftns_sqrd <- log(dfc2$prnt3_dvrg)^2

## *** no logs

## dfc2$inftns <- dfc2$prnt3_dvrg
## dfc2$inftns_sqrd <- dfc2$prnt3_dvrg^2


## ** distinctiveness

## dfc2$disctns <- log(dfc2$cohrt_mean_cos_dists_wtd+0.01)
dfc2$disctns <- log(dfc2$cohrt_mean_non_inf_wtd+0.01)

# dfc2$cohrt_dom <- log(dfc2$volm/dfc2$cohrt_vol_sum)

cohrt_vol_mean <- dfc2$cohrt_vol_sum/dfc2$cohrt_len

## dfc2$cohrt_rel_sz <- log(dfc2$volm/cohrt_vol_mean)
dfc2$cohrt_rel_sz <- log(dfc2$volm/dfc2$cohrt_med)


## ** legitimation
dfc2$leg <- log(dfc2$prnt_plcnt)

## ** density

dfc2$dens_vol <- log(dfc2$cohrt_vol_sum)
dfc2$dens_len <- dfc2$cohrt_len

dfc2$dens_vol_sqrd <- dfc2$dens_vol^2
dfc2$dens_len_sqrd <- dfc2$dens_len^2

## dfc2$dens_vol_sqrd <- log(dfc2$cohrt_vol_sum^2)


## ** size

dfc2$sz <- log(dfc2$volm)
dfc2$new_rlss <- log(dfc2$nbr_rlss_tprd+1)

## ** controls

dfc2$avg_weight_rel_wtd <- dfc2$avg_weight_rel_wtd
ctrl_vars <- c('avg_weight_rel_wtd', 'cos_sims_mean_wtd', 'gnr_gini', 'avg_age', 'sz', 'new_rlss', 'gnr_age2')

## keep cos_sims_mean_wtd as control, would have to come up with proper theory

## ** rel variables

dfc2$tp_id2 <- dfc2$tp_id+1

inf_vars <- c('inftns', 'inftns_sqrd', 'disctns')
dens_vars <- c('dens_vol', 'dens_len', 'dens_vol_sqrd', 'dens_len_sqrd', 'cohrt_rel_sz', 'leg')
# 'cohrt_dom'

## cohrt_domination and cohrt_rel_sz as general controls?
## should check effects

all_vars <- c(inf_vars, dens_vars, ctrl_vars)

## * processing
## ** standardizing


not_scale <- c('gnr_age2')

## dfc3 <- dfc2[0,all_vars]

## for (i in unique(dfc2$tp_id)){
##     dfc3_prep <- as.data.frame(scale(dfc2[which(dfc2$tp_id == i),all_vars[all_vars %!in% not_scale]], center=TRUE,
##                                      scale = apply(dfc2[which(dfc2$tp_id == i),all_vars[all_vars %!in% not_scale]],
##                                                    2, sd, na.rm = T)))
                               
##     dfc3_prep2 <- cbind(dfc3_prep, dfc2[which(dfc2$tp_id == i),c('X', 'tp_id', 'tp_id2', 'event', not_scale)])
##     dfc3 <- rbind(dfc3, dfc3_prep2)
## }

dfc3_prep <- scale(dfc2[,c(all_vars[all_vars %!in% not_scale])])
dfc3 <- cbind(dfc3_prep, dfc2[, c('X', 'tp_id', 'tp_id2', 'event', not_scale)])


## ** descriptives



d <- Dict$new(list(
"inftns" = "Informativeness", 
"inftns_sqrd" = "Informativeness\\textsuperscript{2}",
"disctns" = "Distinctiveness", 
"dens_vol" = "Density (vol)",
"dens_len" = "Density (len)",
"dens_vol_sqrd" = "Density\\textsuperscript{2} (vol)",
"dens_len_sqrd" = "Density\\textsuperscript{2} (len)", 
"cohrt_dom" = "Cohort Domination", 
"cohrt_rel_sz" = "Relative-Cohort Size", 
"leg" = "Legitimacy", 
"avg_weight_rel_wtd" = "Average Tag Weight",
"avg_weight_rel_wtd_sqrd" = "Average Tag Weight\\textsuperscript{2}",
"cos_sims_mean_wtd" = "Song Similarity", 
"gnr_gini" = "Gini", 
"avg_age" = "Average song Age", 
"sz" = "Size", 
"new_rlss" = "New Releases", 
"gnr_age2" = "Genre Age"))




tbl <- t(basicStats(dfc3[,all_vars])[c("Mean", "Median", "Minimum", "Maximum", "nobs", "Stdev", "Skewness", "Kurtosis"),])
colnames(tbl) <- c("Mean", "Median", "Min.", "Max.", "N.Obs", "SD", "Skew", "Kurtosis")
tbl <- round(tbl, 3)
vrbl_names = unlist(lapply(all_vars, function(x){d$get(x)}))

rownames(tbl) <- vrbl_names


latex(tbl,
      title = '',
      caption = 'Summary  Statistics',
      label = 'summaries',
      numeric.dollar=FALSE,
      file = '/home/johannes/Dropbox/gsss/thesis/text/tables/summaries.tex'
      )
## ** correlation



cor_tbl <- cor(dfc3[,all_vars])
cor_tbl[upper.tri(cor_tbl, diag=TRUE)] <- NA
rownames(cor_tbl) <- vrbl_names
colnames(cor_tbl) <- vrbl_names
cor_tbl <- round(cor_tbl, 3)

cor_tbl2 <- cor_tbl[, -ncol(cor_tbl)]


rnms2 <- unlist(lapply(seq(1:nrow(cor_tbl2)), function(x){paste0('(', as.character(x), ') ', rownames(cor_tbl2)[x])}))
cnms2 <- unlist(lapply(seq(1:ncol(cor_tbl2)), function(x){paste0('(', as.character(x), ')')}))

rownames(cor_tbl2) <- rnms2
colnames(cor_tbl2) <- cnms2

latex(cor_tbl2,
      title = '',
      caption = 'Correlation Table',
      label = 'cor_tbl',
      numeric.dollar =  FALSE,
      file = '/home/johannes/Dropbox/gsss/thesis/text/tables/cor_tbl.tex'
      )

## * phreg

dv <- 'Surv(tp_id, tp_id2, event)'
## ** concepts

## exclude new_rlss for current buggy version
## ctrl_vars_bu <- ctrl_vars

## ctrl_vars <- ctrl_vars[ctrl_vars %!in% c('new_rlss')]



ctrl_vars_cbnd <- paste(ctrl_vars, collapse = ' + ')
f_ctrl <- as.formula(paste(c(dv, ctrl_vars_cbnd), collapse = ' ~ '))
fit_ctrl <- phreg(f_ctrl, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
res_ctrl <- phreg_mdlr(fit_ctrl, d, None)


v_inf1 <- paste(c(ctrl_vars_cbnd, 'inftns'), collapse = ' + ')
f_inf1 <- as.formula(paste(c(dv, v_inf1) , collapse = ' ~ ' ))
fit_inf1 <- phreg(f_inf1, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
res_inf1 <- phreg_mdlr(fit_inf1, d, None)
## screenreg(res_inf1)

v_inf2 <- paste(c(ctrl_vars_cbnd, 'inftns', 'inftns_sqrd'), collapse = ' + ')
f_inf2 <- as.formula(paste(c(dv, v_inf2) , collapse = ' ~ ' ))
fit_inf2 <- phreg(f_inf2, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
res_inf2 <- phreg_mdlr(fit_inf2, d, None)
## screenreg(res_inf2)


v_inf3 <- paste(c(ctrl_vars_cbnd, 'disctns'), collapse = ' + ')
f_inf3 <- as.formula(paste(c(dv, v_inf3) , collapse = ' ~ ' ))
fit_inf3 <- phreg(f_inf3, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
res_inf3 <- phreg_mdlr(fit_inf3, d, None)
## screenreg(res_inf3)


v_inf4 <- paste(c(ctrl_vars_cbnd, 'inftns', 'inftns_sqrd', 'disctns'), collapse = ' + ')
f_inf4 <- as.formula(paste(c(dv, v_inf4) , collapse = ' ~ ' ))
fit_inf4 <- phreg(f_inf4, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
res_inf4 <- phreg_mdlr(fit_inf4, d, None)


v_inf5 <- paste(c(ctrl_vars_cbnd, 'inftns', 'inftns_sqrd', 'disctns', 'dens_vol', 'dens_vol_sqrd', 'dens_len', 'dens_len_sqrd', 'leg'), collapse = ' + ')
f_inf5 <- as.formula(paste(c(dv, v_inf5) , collapse = ' ~ ' ))
fit_inf5 <- phreg(f_inf5, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
res_inf5 <- phreg_mdlr(fit_inf5, d, None)



## v_inf6 <- paste(c(ctrl_vars, 'frailty.gamma(factor(X))'), collapse = ' + ')
## v_inf6 <- paste(c(ctrl_vars, 'cluster(factor(X))'), collapse = ' + ')
## f_inf6 <- as.formula(paste(c(dv, v_inf6) , collapse = ' ~ ' ))

## fit_inf6 <- phreg(f_inf6, data=dfc3, cuts = seq(1,26),dist = 'pch')
## res_inf6 <- phreg_mdlr(fit_inf6, d, None)




## screenreg(list(res_ctrl, res_inf1, res_inf2, res_inf3, res_inf4, res_inf5), reorder.coef = c(8:15,1:7))
screenreg(list(res_ctrl, res_inf1, res_inf2, res_inf3, res_inf4, res_inf5))


texreg(list(res_ctrl, res_inf1, res_inf2, res_inf3, res_inf4, res_inf5),
       reorder.coef = c(8:15,1:7),
       label = 'res1',
       caption = 'Impact of conceptual predictors on genre abandonment',
       custom.note = '$^{***}p<0.001$, $^{**}p<0.01$, $^*p<0.05$. Coefficients are logged multiplicative effects on the hazard of abandonment. Standard errors in parantheses', 
       file = '/home/johannes/Dropbox/gsss/thesis/text/tables/res1.tex')

texreg_cleaner('/home/johannes/Dropbox/gsss/thesis/text/tables/res1.tex')



## ** ecology
ctrl_vars_cbnd <- paste(ctrl_vars, collapse = ' + ')
v_dens1 <- paste(c(ctrl_vars_cbnd),collapse = ' + ')
f_dens1 <- as.formula(paste(c(dv, v_dens1), collapse = ' ~ '))
fit_dens1 <- phreg(f_dens1, data=dfc3, cuts= seq(1,28),dist = 'pch')
res_dens1 <- phreg_mdlr(fit_dens1, d, None)


v_dens2 <- paste(c(ctrl_vars_cbnd, 'dens_vol', 'dens_vol_sqrd'),collapse = ' + ')
f_dens2 <- as.formula(paste(c(dv, v_dens2), collapse = ' ~ '))
fit_dens2 <- phreg(f_dens2, data=dfc3, cuts= seq(1,28),dist = 'pch')
res_dens2 <- phreg_mdlr(fit_dens2, d, None)


v_dens3 <- paste(c(ctrl_vars_cbnd, 'dens_len', 'dens_len_sqrd'),collapse = ' + ')
f_dens3 <- as.formula(paste(c(dv, v_dens3), collapse = ' ~ '))
fit_dens3 <- phreg(f_dens3, data=dfc3, cuts= seq(1,28),dist = 'pch')
res_dens3 <- phreg_mdlr(fit_dens3, d, None)

        
v_dens4 <- paste(c(ctrl_vars_cbnd, 'leg'),collapse = ' + ')
f_dens4 <- as.formula(paste(c(dv, v_dens4), collapse = ' ~ '))
fit_dens4 <- phreg(f_dens4, data=dfc3, cuts= seq(1,28),dist = 'pch')
res_dens4 <- phreg_mdlr(fit_dens4, d, None)

v_dens5 <- paste(c(ctrl_vars_cbnd,'dens_vol', 'dens_vol_sqrd', 'dens_len', 'dens_len_sqrd', 'leg'),collapse = ' + ')
f_dens5 <- as.formula(paste(c(dv, v_dens5), collapse = ' ~ '))
fit_dens5 <- phreg(f_dens5, data=dfc3, cuts= seq(1,28),dist = 'pch')
res_dens5 <- phreg_mdlr(fit_dens5, d, None)

v_dens6 <- paste(c(ctrl_vars_cbnd,'dens_vol', 'dens_vol_sqrd', 'dens_len', 'dens_len_sqrd', 'leg', 'inftns', 'inftns_sqrd', 'disctns'),collapse = ' + ')
f_dens6 <- as.formula(paste(c(dv, v_dens6), collapse = ' ~ '))
fit_dens6 <- phreg(f_dens6, data=dfc3, cuts= seq(1,28),dist = 'pch')
res_dens6 <- phreg_mdlr(fit_dens6, d, None)


## pcor(dfc3[,c('dens_vol', 'sz', 'cohrt_dom')])
## from a genres size and the cohort size the cohort domination can be calculated -> one has to go
## cohort_domination: least theoretical relevance


screenreg(list(res_dens1, res_dens2, res_dens3, res_dens4, res_dens5, res_dens6), reorder.coef = c(8:15, 1:7))
screenreg(list(res_dens1, res_dens2, res_dens3, res_dens4, res_dens5, res_dens6))

texreg(list(res_dens1, res_dens2, res_dens3, res_dens4, res_dens5, res_dens6),
       caption = 'Impact of ecological predictors on genre abandonment',
       label = 'res2',
       custom.note = '$^{***}p<0.001$, $^{**}p<0.01$, $^*p<0.05$. Coefficients are logged multiplicative effects on the hazard of abandonment. Standard errors in parantheses',
       custom.model.names = paste('Model', seq(7,12)),
       reorder.coef = c(8:15, 1:7),
       file = '/home/johannes/Dropbox/gsss/thesis/text/tables/res2.tex')


texreg_cleaner('/home/johannes/Dropbox/gsss/thesis/text/tables/res2.tex')

## * comparison



logdiff <- 2*(-538.12 + 545.06)
pchisq(logdiff, df=3 , lower.tail=FALSE)
logdiff <- 2*(-535.38 + 545.06)


logdiff = 2*(-531.41  + 531.38)
logdiff = 2*(3.42)
pchisq(logdiff, df=2, lower.tail=FALSE)

logdiff = 2*(-537.85 + 538.31)
pchisq(logdiff, df=8 , lower.tail=FALSE)

logdiff = -2*(-531.40 + 519.66)

## * figure


## vis_df <- as.data.frame(t(fit_ctrl$hazards))
## names(vis_df) <- 'haz'
## vis_df$tprd <- seq(29)

## pdf('/home/johannes/Dropbox/gsss/thesis/text/figures/hazards.pdf', width = 8, height = 4)
## ggplot(vis_df, aes(x=tprd, y=haz)) +
##     geom_col() +
##     labs(title = 'Hazard rate per time period',
##          y= 'Hazard',
##          x='Time Period') 
## dev.off()

## * tests
fit_fp_test <- frailtyPenal(Surv(tp_id, tp_id2, event) ~ inftns + cluster(X), data = dfc3[dfc3$tp_id > 8,],
                            jointGeneral = FALSE,
                            hazard = 'Piecewise-equi', nb.int = 20)

## ** JM
td.Cox <- coxph(Surv(start, stop, event) ~ drug + CD4,data = aids)

aids.id <- aids[!duplicated(aids$patient), ]

lmeFit.aids <- lme(CD4 ~ obstime + obstime:drug, random = ~ obstime | patient, data = aids)

coxFit.aids <- coxph(Surv(Time, death) ~ drug + gender, data = aids.id, x = TRUE)

jointFit.aids <- jointModel(lmeFit.aids, coxFit.aids, timeVar = "obstime", method = "piecewise-PH-aGH")
summary(jointFit.aids)


## * Brostrom
fit0 <- phreg(Surv(next.ivl, event) ~ parity + ses, dist = 'pch', cuts = 1:13, data=fe)

fe13 <- survSplit(fe, end = 'next.ivl', event = 'event', cut = 1:13, episode = 'years', start = 'start')
## think this is just data transformation into proper format

fe13$years <- as.factor(fe13$years)
fe13$offs <- log(fe13$next.ivl - fe13$start)

fit1 <- glmmML(event ~ parity + ses + years + offset(offs), family = poisson, data = fe13, cluster = id, method = 'ghq', n.points = 9)

fit2 <- glm(event ~ parity + ses + years + offset(offs), family = poisson, data = fe13, method = 'ghq', n.points = 9)

## not clear what offset is doing tho
## seems to be to control for when period is not long enough? 
               
## chapter 6
dfc3$tp_id3 <- as.factor(dfc3$tp_id)
library(glmmML)

fitx <- glmmML(event ~ inftns + inftns_sqrd + disctns + sz + gnr_age2 + tp_id3, family = poisson, data = dfc3, cluster = X, method = 'ghq', n.points = 20)
summary(fitx)
## gives all kind of errors
## doesn't like tp_id3 i think

library(pglm)

fitx2 <- pglm(event ~ inftns + inftns_sqrd + disctns + sz + gnr_age2 + tp_id3,
              family = poisson,
              data = dfc3, index = "X",
              model = 'within')

summary(fitx2)

library(coxme)
fitx3 <- coxme(Surv(tp_id, tp_id2, event) ~ inftns + inftns_sqrd + disctns + sz + gnr_age2 + tp_id3 + (1 | X), data = dfc3)
## mean(fitx3$frail$X[names(fitx3$frail$X)  %in% ded_gnrs4])
## frailty differs ginormously between those that died and those that didn't
## sounds like hell of a lot of unexplained variance

fitx4 <- coxph(Surv(tp_id, tp_id2, event) ~ inftns + inftns_sqrd + disctns + sz + gnr_age2, data = dfc3)
cox.zph(fitx4)

fitx5 <- coxph(f_ctrl, data = dfc3)
cox.zph(fitx5)


fitx6 <- coxph(Surv(tp_id, tp_id2, event) ~ avg_weight_rel_wtd + cos_sims_mean_wtd + 
    gnr_gini + avg_age + sz + new_rlss + inftns + inftns_sqrd + 
    disctns + dens_vol + dens_vol_sqrd + dens_len + dens_len_sqrd + 
    leg  + gnr_age2 + frailty(X), data=dfc3)
screenreg(fitx6)
cox.zph(fitx6)

fitx7 <- coxph(Surv(tp_id, tp_id2, event) ~ inftns + inftns_sqrd + disctns, data = dfc3)
screenreg(fitx7)

res <- cox.zph(fitx6)
plot(res)

## ** replicate phreg with glm

f12 <- fert[fert$parity == 1,]
f12$enter <- 0
f12.split <- survSplit(f12, cut = 1:13, start = 'enter', end = 'next.ivl', event = 'event', episode = 'ivl')

f12.split$offs <- log(f12.split$next.ivl - f12.split$enter)
f12.split$ivl <- as.factor(f12.split$ivl)

fit12.pn <- glm(event ~ offset(offs) + age + year + ses + ivl, family = 'poisson', data = f12.split)

## collapse perids without deaths?
## how would that deal with continuous variables?
## yeahhhh i think not.. especially because equal time frame is crucial for parameter calculation

fc <- age.window(f12.split, c(0,11), surv = c('enter', 'next.ivl', 'event'))
levels(fc$ivl) <- c(0:6, rep('7-11', 7))

fitx <- glm(event ~ offset(offs) + age + year + ses + ivl, family = 'poisson', data = fc)


fitx2 <- glm(event ~ avg_weight_rel_wtd + cos_sims_mean_wtd + gnr_gini + avg_age + sz + new_rlss + gnr_age + tp_id3 + (1 | X), family = 'poisson', data = dfc3)

## wow models are completely equivalent
## i think Brostrom somewhere also has a theoretical argument for it

## might be interesting for diagnostics?




fit_ctrl_wb <- phreg(f_ctrl, data=dfc3, cuts = seq(1,28),dist = 'weibull')
res_ctrl <- phreg_mdlr(fit_ctrl, d, None)

