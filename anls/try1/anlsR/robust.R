## * libs

library(texreg)
library(psych)
library(xtable)
library(Hmisc)
library(collections)
library(fBasics)
library(gmodels)
library(plotrix)

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

phreg_mdlr <- function(mdl, d, imprvmnt, nbr_gnrs){
    vrbls <- mdl$covars
    vrbl_names = unlist(lapply(vrbls, function(x){d$get(x)}))
    coefs <- mdl$coefficients
    vars <- diag(mdl$var)
    ses <- sqrt(vars)
    pvlus <- 2*pnorm(abs(coefs/ses), lower.tail = FALSE)
    
    gof.vlus <- c(nbr_gnrs, mdl$events, mdl$ttr, max(mdl$loglik), mdl$df)
    ## maybe add  model improvement? could do manually later
    ## anova doesn't like 
                  
    gof.names <- c('genres', 'events', 'genre-timeperiods', 'Log-likelihood', 'df')
    

    res <- createTexreg(coef.names = vrbl_names,
                 coef = coefs,
                 se = ses,
                 pvalues = pvlus,
                 gof.names = gof.names,
                 gof = gof.vlus
                 )
    return(res)
}

## * data


dt_prpr <- function(res_dir){

    res_files = list.files(res_dir)
    res_files <- res_files[res_files != 'debug.csv']
    
    dfc <- read.csv(paste0(res_dir, res_files[1]))

    for (i in res_files[2:length(res_files)]){
        res_fl = paste0(res_dir, i)
        dfx <- read.csv(res_fl)
        dfc <- rbind(dfc, dfx)
    }

    harsh_coef <- substr(res_dir, nchar(res_dir)-4, nchar(res_dir)-1)

    empt_endr <- Dict$new(list(
                          '1.25' = 2,
                          '_1.0' = 3,
                          '0.75' = 4))


    EMPT_END <- empt_endr$get(harsh_coef)
    dfc$X <- as.factor(dfc$X)
    dfc <- dfc[order(dfc$X, dfc$tp_id),]


                                        # summarizing
    gnr_inf <- aggregate(dfc$tp_id, list(dfc$X), min)
    gnr_inf2 <- aggregate(dfc$tp_id, list(dfc$X), max)

                                        # defining deds
    ded_gnrs <- gnr_inf2[which(gnr_inf2$x <= max(gnr_inf2$x)-EMPT_END),]$Group.1


    gnr_inf3 <- aggregate(dfc$tp_id, list(dfc$X), length)

    df_gnrs <- cbind(gnr_inf, gnr_inf2$x, gnr_inf3$x)
    names(df_gnrs) <- c('X', 'min', 'max', 'nb_entrs')

    dfc2 <- merge(dfc, df_gnrs, by = 'X')
    dfc2 <- dfc2[order(dfc2$X),]


    ages2 <- unlist(lapply(df_gnrs$nb_entrs, function(x){seq(1:x)}))
    dfc2$gnr_age2 <- ages2


    dfc2$event <- 0
    dfc2$max_tp <- FALSE
    dfc2$max_tp[which(dfc2$tp_id == dfc2$max)] <- TRUE

    dfc2$event[dfc2$X %in% ded_gnrs & dfc2$max_tp ==TRUE] <- 1
    dfc2 <- dfc2[dfc2$tp_id <= max(dfc2$tp_id)-EMPT_END,]


    smpl_ftrs_names <- names(dfc)[unlist(lapply(names(dfc), function(x){grepl('smpl_ftrs', x)}))]

    for (i in smpl_ftrs_names){
        
        new_vrbl_name <- substr(i, 11, 100)
        dfc2[,new_vrbl_name] <- dfc2[,i]
    }
    return(dfc2)
}

dt_fnlzr <- function(dfc2, ctrl_vars){
    dfc2$inftns <- log(dfc2$prnt3_dvrg+0.01)
    dfc2$inftns_sqrd <- log(dfc2$prnt3_dvrg+0.01)^2

    dfc2$disctns <- log(dfc2$cohrt_mean_non_inf_wtd+0.01)

    cohrt_vol_mean <- dfc2$cohrt_vol_sum/dfc2$cohrt_len

    dfc2$cohrt_rel_sz <- log(dfc2$volm/dfc2$cohrt_med)

    dfc2$leg <- log(dfc2$prnt_plcnt)

    dfc2$dens_vol <- log(dfc2$cohrt_vol_sum)
    dfc2$dens_len <- dfc2$cohrt_len

    dfc2$dens_vol_sqrd <- dfc2$dens_vol^2
    dfc2$dens_len_sqrd <- dfc2$dens_len^2

    dfc2$sz <- log(dfc2$volm)
    dfc2$new_rlss <- log(dfc2$nbr_rlss_tprd+1)

    dfc2$avg_weight_rel_wtd <- dfc2$avg_weight_rel_wtd
    ## ctrl_vars <- c('avg_weight_rel_wtd', 'cos_sims_mean_wtd', 'gnr_gini', 'avg_age', 'sz', 'new_rlss', 'gnr_age2')
    dfc2$mean_prnt_kld_wtd <- log(dfc2$mean_prnt_kld_wtd+0.01)


    dfc2$tp_id2 <- dfc2$tp_id+1

    inf_vars <- c('inftns', 'inftns_sqrd', 'disctns')
    dens_vars <- c('dens_vol', 'dens_len', 'dens_vol_sqrd', 'dens_len_sqrd', 'cohrt_rel_sz', 'leg')

    all_vars <- c(inf_vars, dens_vars, ctrl_vars)

    not_scale <- c('gnr_age2')

    dfc3_prep <- scale(dfc2[,c(all_vars[all_vars %!in% not_scale])])
    dfc3 <- cbind(dfc3_prep, dfc2[, c('X', 'tp_id', 'tp_id2', 'event', not_scale)])


    return(dfc3)
}

mdl_rnr <- function(dfc3, ctrl_vars){

    
    d <- Dict$new(list(
                  "inftns" = "Informativeness", 
                  "inftns_sqrd" = "Informativeness\\textsuperscript{2}",
                  "disctns" = "Distinctiveness", 
                  "dens_vol" = "Density (vol)",
                  "dens_len" = "Density (len)",
                  "mean_prnt_kld_wtd" = "Parent similarity",
                  "cohrt_ovlp" = "Cohort Overlap",
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

    inf_vars <- c('inftns', 'inftns_sqrd', 'disctns')
    dens_vars <- c('dens_vol', 'dens_len', 'dens_vol_sqrd', 'dens_len_sqrd', 'cohrt_rel_sz', 'leg')

    all_vars <- c(inf_vars, dens_vars, ctrl_vars)

    tbl <- t(basicStats(dfc3[,all_vars])[c("Mean", "Median", "Minimum", "Maximum"),])
    colnames(tbl) <- c("Mean", "Median", "Min.", "Max.")
    tbl <- round(tbl, 3)
    vrbl_names = unlist(lapply(all_vars, function(x){d$get(x)}))
    rownames(tbl) <- vrbl_names

    cor_tbl <- cor(dfc3[,all_vars])
    cor_tbl[upper.tri(cor_tbl, diag=TRUE)] <- NA
    rownames(cor_tbl) <- vrbl_names
    colnames(cor_tbl) <- vrbl_names
    cor_tbl <- round(cor_tbl, 3)

    cor_tbl2 <- cor_tbl[, -ncol(cor_tbl)]


    rnms2 <- unlist(lapply(seq(1:nrow(cor_tbl2)),
                           function(x){paste0('(', as.character(x), ') ', rownames(cor_tbl2)[x])}))
    cnms2 <- unlist(lapply(seq(1:ncol(cor_tbl2)), function(x){paste0('(', as.character(x), ')')}))

    rownames(cor_tbl2) <- rnms2
    colnames(cor_tbl2) <- cnms2




    dv <- 'Surv(tp_id, tp_id2, event)'
    ctrl_vars_cbnd <- paste(ctrl_vars, collapse = ' + ')
    v_inf5 <- paste(c(ctrl_vars_cbnd, 'inftns', 'inftns_sqrd', 'disctns', 'dens_vol', 'dens_vol_sqrd', 'dens_len', 'dens_len_sqrd', 'leg'), collapse = ' + ')
    f_inf5 <- as.formula(paste(c(dv, v_inf5) , collapse = ' ~ ' ))
    fit_inf5 <- phreg(f_inf5, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
    res_inf5 <- phreg_mdlr(fit_inf5, d, None, len(unique(dfc3$X)))

 
    ## fitx_ph <- coxph(f_inf5, data = dfc3)
    ## print(screenreg(fitx5))
    ## print(cox.zph(fitx5))

    frail_vars <- paste(c(v_inf5, 'frailty(X)'), collapse = ' + ')
    frail_f <- as.formula(paste(c(dv, frail_vars), collapse = ' ~ '))
    fit_frailty <- coxph(frail_f, data = dfc3)


    return(list(res_inf5, fit_frailty, tbl, cor_tbl2))
}


proc_func <- function(res_dir){

    print(res_dir)

    ctrl_vars <- c('avg_weight_rel_wtd', 'cos_sims_mean_wtd', 'gnr_gini', 'avg_age', 'sz', 'new_rlss', 'gnr_age2')
    dfc2 <- dt_prpr(res_dir)

    dfc3 <- dt_fnlzr(dfc2, ctrl_vars)

    res <- mdl_rnr(dfc3, ctrl_vars)

    return(res)
}

## * actually running

res_dirs <- c(
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_0.75_tp_0.75/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_1.0_tp_0.75/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_1.25_tp_0.75/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_0.75_tp_1.0/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_1.0_tp_1.0/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_1.25_tp_1.0/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_0.75_tp_1.25/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_1.0_tp_1.25/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust_v2/harsh_1.25_tp_1.25/'
)



all_res <- lapply(res_dirs, proc_func)

pch_res <- lapply(all_res, function(x){x[[1]]})
frail_res <- lapply(all_res, function(x){x[[2]]})
sum_tbls  <- lapply(all_res, function(x){x[[3]]})
cor_tbls  <- lapply(all_res, function(x){x[[4]]})


## * summary stats

## ** descriptives
arr <- array( unlist(sum_tbls) , c(16,4,9))
arr_means <- apply(arr, 1:2, mean)
arr_ses <- apply(arr, 1:2, std.error)

sum_trs = list()

for (i in 1:4){
    print(i)
    sum_tr <- createTexreg(coef = arr_means[,i],
                           coef.names = rownames(sum_tbls[[1]]),
                           ci.up = arr_means[,i] + 1.96 * arr_ses[,i],
                           ci.low = arr_means[,i] - 1.96 * arr_ses[,i])
    
    ## sum_trs <- c(sum_trs, sum_tr)
    sum_trs[i] <- sum_tr
}

p_lists = lapply(1:4, function(x){rep(0.5, 16)})
se_lists = lapply(1:4, function(x){rep(10, 16)})
    
texreg(sum_trs, custom.model.names = c('Mean', 'Median', 'Min.', 'Max.'),
       ci.test = NULL,
       caption = 'Summary Statistics',
       label = 'summaries',
       custom.note = '95 \\% CI of the 9 models in square brackets',       
       file = '~/Dropbox/gsss/thesis/text/tables/summaries_mult.tex', 
       use.packages = FALSE)

texreg_cleaner('~/Dropbox/gsss/thesis/text/tables/summaries_mult.tex')     

## ** correlation
arr_cor <- array(unlist(cor_tbls), c(16, 15, 9))
arr_cor_means <- apply(arr_cor, 1:2, mean)
arr_cor_ses <- apply(arr_cor, 1:2, std.error)

cor_trs = list()

for (i in 1:15){
    print(i)
    cor_tr <- createTexreg(coef = arr_cor_means[,i],
                           coef.names = rownames(cor_tbls[[1]]),
                           ci.up = arr_cor_means[,i] + 1.96 * arr_cor_ses[,i],
                           ci.low = arr_cor_means[,i] - 1.96 * arr_cor_ses[,i])
    
    ## sum_trs <- c(sum_trs, sum_tr)
    cor_trs[i] <- cor_tr
}

texreg(cor_trs, ci.test = NULL, custom.model.names = colnames(cor_tbls[[1]]),
       sideways = TRUE, file = '~/Dropbox/gsss/thesis/text/tables/cors_mult.tex',
       use.packages = FALSE, scalebox = 0.65,
       caption = 'Correlation Table',
       label = 'cor_tbl',
       custom.note = '95 \\% CI of the 9 models in square brackets'
       )

texreg_cleaner('~/Dropbox/gsss/thesis/text/tables/cors_mult.tex')





screenreg(pch_res)
## screenreg(ph_res)
screenreg(frail_res)


## * meta analysis

## https://stats.stackexchange.com/questions/5586/average-effect-of-coefficients-across-multiple-linear-models
## The standard error of the average then will be the square root from the average of the squares of the standard errors.


## lapply(frail_res, function(x){x[[1]]@'se(coef)'^2})

ses <- lapply(pch_res, function(x){x@se^2})
se_mat <- matrix(unlist(ses), ncol = 9)
se_avg <- sqrt(rowMeans(se_mat))

coefs <- lapply(pch_res, function(x){x@coef})
coefs_mean <- rowMeans(matrix(unlist(coefs), ncol = 9))

pvlus <- 2*pnorm(abs(coefs_mean/se_avg), lower.tail = FALSE)

meta_mdl <- createTexreg(coef.names = pch_res[1][[1]]@coef.names, coef = coefs_mean, se = se_avg, pvalues = pvlus)
texreg(meta_mdl, custom.model.names = c('Genre Abandonment'),
       dcolumn = TRUE, use.packages = FALSE,
       caption = "Summary Model",
       label = 'res_sum',
       custom.note = '\\parbox{.4\\linewidth}{\\vspace{2pt}%stars. \\\\Coefficients (averaged across models) are logged multiplicative effects on the hazard of abandonment. Standard errors in parentheses}.',
       file = '~/Dropbox/gsss/thesis/text/tables/res_sum.tex',
       table = FALSE)



## * export

## ** pch

mod_names <- rep(c('lenient', 'medium', 'strict'), 3)

texreg3(pch_res,
        custom.model.names = mod_names,
        col.groups = list('12 weeks' = 1:3, '16 weeks' = 4:6, '20 weeks' = 7:9) ,
        file = '~/Dropbox/gsss/thesis/text/tables/pch_res.tex',
        caption = 'Piecewise Constant Exponential Hazard Models',
        label = 'pch',
        dcolumn = TRUE,
        sideways = TRUE,
        use.pa tckages = FALSE)

## texreg_cleaner('~/Dropbox/gsss/thesis/text/tables/mult_test.tex')

## ** frailty
mod_names <- rep(c('lenient', 'medium', 'strict'), 3)

texreg3(frail_res,
        custom.model.names = mod_names,
        col.groups = list('12 weeks' = 1:3, '16 weeks' = 4:6, '20 weeks' = 7:9) ,
        file = '~/Dropbox/gsss/thesis/text/tables/frail_res.tex',
        caption = 'Proportional Hazard Models with individual frailty',
        label = 'frailty',
        dcolumn = TRUE,
        custom.coef.names = pch_res[1][[1]]@coef.names,
        sideways = TRUE,
        use.packages = FALSE)




## * poisson test
dfc3$tp_id3 <- as.factor(dfc3$tp_id)
fitx <- glm(event ~ avg_weight_rel_wtd + cos_sims_mean_wtd + 
    gnr_gini + avg_age + sz + new_rlss + gnr_age2 + inftns + 
    inftns_sqrd + disctns + dens_vol + dens_vol_sqrd + dens_len + 
    dens_len_sqrd + leg +tp_id3, data = dfc3, family = 'poisson')


for (i in ctrl_vars){
    print(i)
    cor(dfc2$mean_prnt_kld_wtd, dfc2[,i])
    }

lapply(ctrl_vars, function(x){cor(dfc2$mean_prnt_kld_wtd, dfc3[,x])})



## * debug shit

debugx <- read.csv('/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust/harsh_0.75_tp_0.75/debug.csv', header = FALSE)
## head(debugx)

debugx2 <- debugx[debugx$V2 < 15,]
write.table(debugx2, '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/robust/harsh_0.75_tp_0.75/debug2.csv', row.names=FALSE, quote = FALSE, col.names = FALSE, sep = ',')

