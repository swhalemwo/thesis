library(reshape2)
library(ggplot2)

df <- read.csv('/home/johannes/Dropbox/gsss/thesis/anls/try1/results/t1.csv', header=FALSE)

names(df) <- c('tp', 'tag', 'pop')
df$tp <- factor(df$tp)

unqs <- unique(df$tag)
unqs2 <- sample(unqs,6)


dfc <- dcast(df, tag~tp)
dfcs <- dfc[1:5,]

apply(dfcs[2:14], 1, function(x){sum(is.na(x))})
apply(dfcs[15:26], 1, function(x){sum(is.na(x))})

dfc$mis.p1 <- apply(dfc[2:14], 1, function(x){sum(is.na(x))})
dfc$mis.p2 <- apply(dfc[15:26], 1, function(x){sum(is.na(x))})

dfc$mis.rto <- dfc$mis.p1/dfc$mis.p2

dfc[which(is.nan(dfc$mis.rto)),]$mis.rto <- 1

## dfc$mis.rto[is.nan(dfc$mis.rto)] <- 1

diers <- dfc$tag[(which(dfc$mis.rto < 0.5))]
unqs2 <- sample(diers,6)



subdf <- df[which(df$tag %in% unqs2),]
subdf$tag <- factor(subdf$tag)


ggplot(subdf, aes(x=tp, y=pop, group=tag))+
    geom_line(aes(linetype=tag, color=tag),size=1)




dfc <- dcast(df, tag~tp)

subdf <- dfc[1:10,1:26]


