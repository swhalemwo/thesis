library(PerformanceAnalytics)


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
