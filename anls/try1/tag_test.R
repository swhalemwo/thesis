library(RClickhouse)



con <- DBI::dbConnect(RClickhouse::clickhouse(), host="localhost", password='anudora', database='frrl')

DBI::dbGetQuery(con, 'show databases')
df <- DBI::dbGetQuery(con, 'select count(*) from frrl.tags')

df <- DBI::dbGetQuery(con, qry)

qry="
select tag,
avg(weight) as avg_weight,
count(tag) as cnt,
quantileExact(0.25)(weight) as q25,
quantileExact(0.5)(weight) as q50,
quantileExact(0.75)(weight) as q75
from frrl.tags where weight > 10 group by tag having count(tag) > 50 order by avg(weight) desc
"

ggplot(df, aes(x=avg_weight, y=log(cnt)))+
    geom_point(aes(size=q50*0.1))
    ## geom_jitter(aes(size=cnt))
