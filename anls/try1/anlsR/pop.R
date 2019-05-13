library(reshape2)
library(ggplot2)

df <- read.csv('/home/johannes/Dropbox/gsss/thesis/anls/try1/results/t1.csv', header=FALSE)

names(df) <- c('tp', 'tag', 'pop')
df$tp <- factor(df$tp)

unqs <- unique(df$tag)
unqs2 <- sample(unqs,6)


dfc <- dcast(df, tag~tp)
dfcs <- dfc[1:5,]

## apply(dfcs[2:14], 1, function(x){sum(is.na(x))})
## apply(dfcs[15:26], 1, function(x){sum(is.na(x))})

## dfc$miz.p1 <- apply(dfc[2:14], 1, function(x){sum(is.na(x))})
## dfc$miz.p2 <- apply(dfc[15:26], 1, function(x){sum(is.na(x))})

last.ent.finder <- function(x){
    real.ent = 0
    na.ent = 0

    for (i in seq(length(x))){
        ## print(i)
        if (is.na(x[i])){
            na.ent = i
        } else {
            real.ent = i
        }
    }
    ## print(c(real.ent, na.ent))
    
    return(real.ent)
}

dfc$last.ent <- apply(dfc[2:26], 1, last.ent.finder)
dfc$missing <- apply(dfc[2:26], 1, function(x){sum(is.na(x))})

            
## dfc$mis.rto <- dfc$miz.p1/dfc$miz.p2
## hist(dfc$mis.rto, breaks=50)

## need to rework that function to identity genres

## dfc[which(is.nan(dfc$mis.rto)),]$mis.rto <- 1

## dfc$mis.rto[is.nan(dfc$mis.rto)] <- 1

diers <- dfc$tag[(which(dfc$last.ent < 20))]

diers <- dfc$tag[(which(dfc$last.ent < 20 & dfc$missing < 15))]


unqs2 <- diers[1:6]
unqs2 <- sample(diers,6)



subdf <- df[which(df$tag %in% unqs2),]
subdf$tag <- factor(subdf$tag)


ggplot(subdf, aes(x=tp, y=pop, group=tag))+
    geom_line(aes(linetype=tag),size=1)
    + theme_apa()



dfc <- dcast(df, tag~tp)

subdf <- dfc[1:10,1:26]



cleaning
- lower case
- train model on musical texts
- word embedding
- word2vec

