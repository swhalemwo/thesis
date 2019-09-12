## * libs

library(texreg)
library(psych)
library(xtable)
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

## * data
res_dirs <- c(
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/harsh_narrow/',
    '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/harsh_wide/'
)


dt_prpr <- function(res_dir){

    res_files = list.files(res_dir)
    res_files <- res_files[res_files != 'debug.csv']
    
    dfc <- read.csv(paste0(res_dir, res_files[1]))

    for (i in res_files[2:length(res_files)]){
        res_fl = paste0(res_dir, i)
        dfx <- read.csv(res_fl)
        dfc <- rbind(dfc, dfx)
    }

    EMPT_END <- 4
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
    dfc2$inftns <- log(dfc2$prnt3_dvrg)
    dfc2$inftns_sqrd <- log(dfc2$prnt3_dvrg)^2

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

    dv <- 'Surv(tp_id, tp_id2, event)'
    ctrl_vars_cbnd <- paste(ctrl_vars, collapse = ' + ')
    v_inf5 <- paste(c(ctrl_vars_cbnd, 'inftns', 'inftns_sqrd', 'disctns', 'dens_vol', 'dens_vol_sqrd', 'dens_len', 'dens_len_sqrd', 'leg'), collapse = ' + ')
    f_inf5 <- as.formula(paste(c(dv, v_inf5) , collapse = ' ~ ' ))
    fit_inf5 <- phreg(f_inf5, data=dfc3, cuts = seq(min(dfc3$tp_id)+1,max(dfc3$tp_id)),dist = 'pch')
    res_inf5 <- phreg_mdlr(fit_inf5, d, None)

    return(res_inf5)
}


proc_func <- function(res_dir){

    ctrl_vars <- c('avg_weight_rel_wtd', 'cos_sims_mean_wtd', 'gnr_gini', 'avg_age', 'sz', 'new_rlss', 'gnr_age2')
    dfc2 <- dt_prpr(res_dir)

    dfc3 <- dt_fnlzr(dfc2, ctrl_vars)

    res <- mdl_rnr(dfc3, ctrl_vars)

    return(res)
}

all_res <- lapply(res_dirs, proc_func)

screenreg(all_res)


# seems that 

