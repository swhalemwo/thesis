library(ggplot2)

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, survival, ggfortify, survminer, plotly, gridExtra, 
               Epi, KMsurv, gnm, cmprsk, mstate, flexsurv, splines, epitools, 
               eha, shiny, ctqr, scales)

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


## * lets merge data together

res_dir = '/home/johannes/Dropbox/gsss/thesis/anls/try1/results/24_weeks/'
res_files = list.files(res_dir)

dfc <- read.csv(paste0(res_dir, res_files[1]))

for (i in res_files[2:length(res_files)]){
    res_fl = paste0(res_dir, i)
    dfx <- read.csv(res_fl)
    dfc <- rbind(dfc, dfx)
}


## just rbind everything together
## check consistency with length of range: more range covered than periods -> inconsistent
## for those that are consistent: if max range < max(range for unit) -> failure at that time
## not sure if that lags properly but whatever


dfc$X <- as.factor(dfc$X)
dfc <- dfc[order(dfc$X, dfc$tp_id),]

dfc$event <- 0

gnr_inf <- aggregate(dfc$tp_id, list(dfc$X), min)
gnr_inf2 <- aggregate(dfc$tp_id, list(dfc$X), max)
## 364 failed
## that's not nothing but not nearly as much as i wish

ded_gnrs <- gnr_inf2[which(gnr_inf2$x < 14),]$Group.1

gnr_inf3 <- aggregate(dfc$tp_id, list(dfc$X), length)

## see which are complete

df_gnrs <- cbind(gnr_inf, gnr_inf2$x, gnr_inf3$x)
names(df_gnrs) <- c('gnr', 'min', 'max', 'nb_entrs')

df_gnrs$max2 <- df_gnrs$max +1
df_gnrs$chk <- df_gnrs$max2 - df_gnrs$nb_entrs

## cheese genres due to holes like swiss cheese
cheese_gnrs <- df_gnrs[which(df_gnrs$min != df_gnrs$chk),'gnr']

length(which(ded_gnrs %!in% cheese_gnrs))
## i mean still have 244 complete ded gnrs

dfc2 <- dfc[which(dfc$X %!in% cheese_gnrs),]

ded_gnrs2 <- unique(dfc2$X)[which(unique(dfc2$X) %in% ded_gnrs)]

## i = 'Bleach'
for (i in ded_gnrs2){

    dfx <- dfc2[which(dfc2$X ==i),]
    max_tp <- max(dfx$tp)

    dfc2[which(dfc2$X == i & dfc2$tp_id == max_tp),]$event <- 1
    ## print(c(i, dim(dfx)))
}

deds <- aggregate(dfc2$event, list(dfc2$tp_id), sum)
barplot(deds[,2])

## ** reg tes
coxph(formula = Surv(tstart, tstop, infect) ~ treat + inherit + steroids +cluster(id), data = newcgd)

dfc2$tp_id2 <- dfc2$tp_id+1

dfc3 <- dfc2[-which(dfc2$prnt_odg_wtd==Inf),]

dfc4 <- dfc2[-10,]

fit <- coxph(Surv(tp_id, tp_id2, event) ~ avg_weight_rel + cohrt_pct_inf + sz_raw + prnt3_dvrg + prnt_odg + prnt_odg_wtd + spngns + spngns_std + cluster(X), data= dfc3)
## size is fairly obvious
## prnt3_dvrg is more interesting tbh: more informative -> more likely to die

fit2 <- coxph(Surv(tp_id, tp_id2, event) ~ avg_weight_rel + cohrt_pct_inf + cohrt_mean_non_inf + 
                 sz_raw + prnt3_dvrg + prnt_odg + prnt_odg_wtd +
                 spngns + spngns_std + cluster(X), data= dfc3)
## including cohrt_mean_non_inf and deleting a bunch -> cohrt_pct_inf suddenly significant, prnt3 dvrg not anymore


fit3 <- coxph(Surv(tp_id, tp_id2, event) ~ cohrt_pct_inf + prnt3_dvrg + I(prnt3_dvrg^2)+
                 sz_raw +  prnt_odg + prnt_odg_wtd +
                 spngns + spngns_std + cluster(X), data= dfc3)


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




