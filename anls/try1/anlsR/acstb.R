library(PerformanceAnalytics)

## * inspect variable distributions

df <- read.csv('/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/acstbrnz.csv', header=FALSE)


names(df) <- c('id', 'dncblt', 'gender', 'timb_brt', 'tonal', 'voice', 'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy', 'mood_party', 'mood_relaxed', 'mood_sad', 'gnr_dm_alternative', 'gnr_dm_blues', 'gnr_dm_electronic', 'gnr_dm_folkcountry', 'gnr_dm_funksoulrnb', 'gnr_dm_jazz', 'gnr_dm_pop', 'gnr_dm_raphiphop', 'gnr_dm_rock', 'gnr_rm_cla', 'gnr_rm_dan', 'gnr_rm_hip', 'gnr_rm_jaz', 'gnr_rm_pop', 'gnr_rm_rhy', 'gnr_rm_roc', 'gnr_rm_spe', 'gnr_tza_blu', 'gnr_tza_cla', 'gnr_tza_cou', 'gnr_tza_dis', 'gnr_tza_hip', 'gnr_tza_jaz', 'gnr_tza_met', 'gnr_tza_pop', 'gnr_tza_reg', 'gnr_tza_roc', 'mirex_Cluster1', 'mirex_Cluster2', 'mirex_Cluster3', 'mirex_Cluster4', 'mirex_Cluster5', 'length', 'label', 'lang', 'rl_type', 'rls_cri')


chart.Correlation(df[,c('dncblt', 'gender', 'timb_brt', 'tonal', 'voice')])

chart.Correlation(df[,c('mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy', 'mood_party', 'mood_relaxed', 'mood_sad')])

chart.Correlation(df[,c('gnr_dm_alternative', 'gnr_dm_blues', 'gnr_dm_electronic', 'gnr_dm_folkcountry', 'gnr_dm_funksoulrnb', 'gnr_dm_jazz', 'gnr_dm_pop', 'gnr_dm_raphiphop', 'gnr_dm_rock')])
## seems to be mostly electronic LOL

sum(apply(df[,c('gnr_dm_alternative', 'gnr_dm_blues', 'gnr_dm_electronic', 'gnr_dm_folkcountry', 'gnr_dm_funksoulrnb', 'gnr_dm_jazz', 'gnr_dm_pop', 'gnr_dm_raphiphop', 'gnr_dm_rock')], 2, mean))
## yup

chart.Correlation(df[,c('gnr_rm_cla', 'gnr_rm_dan', 'gnr_rm_hip', 'gnr_rm_jaz', 'gnr_rm_pop', 'gnr_rm_rhy', 'gnr_rm_roc', 'gnr_rm_spe')])

apply(df[,c('gnr_rm_cla', 'gnr_rm_dan', 'gnr_rm_hip', 'gnr_rm_jaz', 'gnr_rm_pop', 'gnr_rm_rhy', 'gnr_rm_roc', 'gnr_rm_spe')], 2, mean)
## doesn't have electronic hmm -> bad
## thinks it's mostly rock (35%), rhythm (15%), pop (17%)


chart.Correlation(df[,c('gnr_tza_blu', 'gnr_tza_cla', 'gnr_tza_cou', 'gnr_tza_dis', 'gnr_tza_hip', 'gnr_tza_jaz', 'gnr_tza_met', 'gnr_tza_pop', 'gnr_tza_reg', 'gnr_tza_roc')])
## weird distributions
## not bimodal, but mode is spike at varying positions between 0 and 0.4

barplot(apply(df[,c('gnr_tza_blu', 'gnr_tza_cla', 'gnr_tza_cou', 'gnr_tza_dis', 'gnr_tza_hip', 'gnr_tza_jaz', 'gnr_tza_met', 'gnr_tza_pop', 'gnr_tza_reg', 'gnr_tza_roc')], 2, mean))
## thinks jazz is most there, followed by hip hop and rock


chart.Correlation(df[,c('mirex_Cluster1', 'mirex_Cluster2', 'mirex_Cluster3', 'mirex_Cluster4', 'mirex_Cluster5')])
## similarly weird distributions as for tzanetakis genres
## genres are spikes at at other places than the extremes

## according to the MBZ website, the genres have all pretty shitty recognition(dortmund 60,tzanetakis 75, mirex 57)
## except rosamerica 87
## but doesn't have electronic???

## just use those with 90%+?
## danceability: 92
## gender: 87
## timbre: 94
## tonal_atonal: 97
## voice_instrumental: 93
2
## mood acoustic: 92
## mood aggressive: 97
## mood electronic: 87
## mood happy: 83
## mood party: 88
## mood relaxed: 93
## mood sad: 87

barplot(apply(df[,c('mirex_Cluster1', 'mirex_Cluster2', 'mirex_Cluster3', 'mirex_Cluster4', 'mirex_Cluster5')], 2, mean))


ggplot(df, aes(x=timb_brt, y=tonal)) +
    geom_point(size=0.05)



## bimodal distributions for:
##                           - dncblt
## - tonal
## - timbre
## - voice

## weird:
##     - gender: peak in middle
##     - mood_acoustic: has step in middle: results in weird shapes of fitted line

## uni? modal for the moods
## less skewed/less steep for happy

## correlations:
##     - tim_brt, tonal: 0.63
## - tim_brt, mood_electronic: -0.59
## - mood_aggressive, mood_party: 0.69
## - mood_aggressive, mood_relaxed: -0.71
## - mood party, mood relaxed: -0.74
## - mood acoustic, mood_sad: 0.8


## * test niches with example, most basic case

ggplot(df, aes(x=timb_brt, y=tonal)) +
    geom_point(size=0.05)

library(RClickhouse)

con <- DBI::dbConnect(RClickhouse::clickhouse(), host="localhost", password='anudora', database='frrl')

DBI::dbGetQuery(con, 'show databases')
DBI::dbGetQuery(con, 'show tables')

gnrs <- c('black metal', 'shoegaze', 'punk', 'classic', 'rap', 'post-grunge')

qryer <- function(gnr){
    ## qry = paste0("select * from frrl.tag_sums where tag = '", gnr, "' and rel_weight > 0.03")

    qry = paste0("select mbid, tag, rel_weight, cnt from frrl.tag_sums ",
                 "join (select mbid, cnt from frrl.song_info3) using(mbid) where tag='", gnr, "' and rel_weight > 0.03")

    df.tags <- DBI::dbGetQuery(con, qry)

    print(dim(df.tags))
    df.tags2 <- df.tags[which(df.tags$mbid %in% df$id),]
    print(dim(df.tags2))
    return(df.tags2)
}

## library(raster)
library(radiant.data)

df.tags <- qryer(gnr)


volumer <- function(df.tags, gnr){
    ## df[,gnr] <- 0
    ## df[,gnr][which(df$id %in% df.tags$mbid)] <- 1
    ## df[,gnr][which(df$id %in% df.tags$mbid)] <- df.tags2$rel_weight

    dfx <- merge(df, df.tags, by.x ='id', by.y='mbid', all=TRUE)
    
    dims = c("dncblt", "gender", "timb_brt", "tonal", "voice", 'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy', 'mood_party', 'mood_relaxed', 'mood_sad')

    ttl_vol = c()
    tops <- c()
    bots <- c()
        
    for (dimx in dims){
        meanx <- mean(dfx[which(dfx$tag == gnr),dimx])
        cnts <- dfx[which(dfx$tag==gnr),'cnt']
        rweits <- dfx[which(dfx$tag==gnr),'rel_weight']

        ## hm those with high playcounts have low relweights, those with high relweights have low cnts
        ## makes sense tbh
        ## cbnd <- cnts * rweits
        ## cbnd <- sqrt(cnts) * rweits

        corx <- cor(cnts, rweits)
        ## cbnd correlates stronger with cnts than rweits

        w.meanx  <- weighted.mean(dfx[which(dfx$tag==gnr),dimx], rweits)
        ## need to decide what to use
        ## unweighted seems to be a bit bigger in ~75% of cases
        ## print(c(meanx, w.meanx))
        meanx <- w.meanx
        
        ## lel basically same as non weighted mean
        ## is theoretical question: what defines a genre: is it a song that is played a lot, but only has few tags?
        ## or is it a song that isn't played a lot but is very strongly associated with it?
        ## hmm would say second one, rweits
        ## maybe sqrt(cnts)

        sdx = sd(df[which(dfx$tag==gnr),dimx])
        w.sdx <- weighted.sd(dfx[which(dfx$tag==gnr),dimx], rweits)
        ## print(c(sdx, w.sdx))

        ## need to decide what to use
        sdx <- w.sdx
            
        top = meanx + 1.5*sdx
        bot = meanx - 1.5*sdx

        tops <- c(tops, top)
        bots <- c(bots, bot)

        rngx = top-bot
        ttl_vol <- c(ttl_vol, rngx)
    }

    pruct = prod(ttl_vol)

    print(c(gnr, round(pruct,3)))

    ## see how many other songs are in niche
    dfx2 <- dfx

    for (i in dims){
        cx <- which(dims %in% i)
        topx <- tops[cx]
        botx <- bots[cx]

        dfx2 <- dfx2[which(dfx2[,i] > botx & dfx2[,i] < topx),]
        
    }
    print(c(dim(df.tags)[1],dim(dfx2)[1], dim(df.tags)[1]/dim(dfx2)[1]))
    ## NEED TO GET playcount of all songs
}


for (i in gnrs){
    df.tags = qryer(i)
    volumer(df.tags, i)
}
## neato
## add: weighted mean/sd?
## weigh by
## - plcnt
## - weight
## product?

## if i weight by plcnt, do i still get probability distribution?



limit 10

## qry = "select * from frrl.tag_sums where tag = 'pop rock' and rel_weight > 0.03"
## should get playcount here as well
df.tags <- DBI::dbGetQuery(con, qry)

df.tags2 <- df.tags[which(df.tags$mbid %in% df$id),]

df$pop_rock <- 0
df$pop_rock[which(df$id %in% df.tags2$mbid)] <- 1

df$pop_rock <- factor(df$pop_rock)

ggplot(df, aes(x=dncblt, y=tonal)) +
    geom_point(aes(color=pop_rock), size=0.1)
## size=as.numeric(pop_rock)))

ggplot(df[which(df$pop_rock==1),], aes(x=dncblt, y=tonal)) +
    geom_jitter(width=0.05, height=0.05) + 
    geom_rect(mapping=aes(xmin=0.25, xmax=0.5, ymin=0.25, ymax=0.5), alpha=0.005)



gnr = 'pop_rock'


for (dimx in dims){
    print(mean(df[which(df[,gnr]==1),dimx]))
    print(sd(df[which(df[,gnr]==1),dimx]))
}







## basically cover entire space




mt2 <- mtcars

mt2$brand <- rownames(mt2)
mt2$brand2 <- tolower(mt2$brand)

mt2[which(mt2$cyl==4),]$brand <- tolower(mt2[which(mt2$cyl==4),]$brand)


names(mt2) <- toupper(names(mt2))
names(mt2) <- tolower(names(mt2))


str_cols = c(11,12)


lowerer <- function(v){
    if(is.character(v) ==TRUE) {
        print(typeof(v))
        
        v2 <- tolower(v)
        ## print('ee')
    } else {
        v2  <- v
        print('dd')
    }
    return(v2)}

mt4 <- apply(mt2[,str_cols], 2, lowerer)
