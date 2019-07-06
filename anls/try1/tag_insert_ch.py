import csv
from clickhouse_driver import Client



client = Client(host='localhost', password='anudora', database='frrl')


client.execute('drop table tags')
client.execute("""create table tags
(mbid String, tag String, weight Integer)
engine=MergeTree() partition by weight order by tuple()""")


# tag_file = "/home/johannes/Dropbox/gsss/thesis/anls/try1/add_data/tag_chunks/test_split2/1_tags.csv"
tag_file = '/home/johannes/mega/gsss/thesis/remotes/ttl/ttl_tags4.csv'

# works now 
# subsequent reading in might work in one go when de-nullified with tr and large try statement


batch=[]
c=0
c2 = 0
with open(tag_file, 'r') as fi:
    rdr = csv.reader(fi)
    for row in rdr:
        try:
            batch.append([row[0], row[1], int(row[2])])
            c+=1
            c2+=1
            if c == 25000:
                # break
                client.execute('insert into tags values', batch)
                c=0
                batch=[]
                print(c2)

        except:
            print('lel')


# * Update weights

client.execute("drop table tag_sums")

client.execute("""CREATE TABLE tag_sums (
mbid String, 
tag String, 
weight Int32, 
ttl_weight Int32)
engine=MergeTree() 
partition by weight 
order by tuple()""")

client.execute("""INSERT INTO tag_sums
SELECT * FROM tags 
JOIN (SELECT mbid, sum(weight) FROM tags WHERE weight > 10 GROUP BY mbid) USING (mbid)""")

client.execute("ALTER TABLE tag_sums ADD COLUMN rel_weight Float32")

client.execute("ALTER TABLE tag_sums UPDATE rel_weight = divide(weight,ttl_weight) WHERE weight > 0")


## can play with parameters to get nice number of tags
select count(*) from (
    select tag, count(tag) from tags where weight > 10 group by tag having count(tag) > 30 order by count(tag) desc

select count(*) from (
    select tag, count(tag) from tag_sums where rel_weight > 0.05 group by tag having count(tag) > 30)

# * investigating problems with NULL byte

    # c2 = 25024374
    # c = 24374
    # exactly 25m in db atm

    # sed -n '25024372p;25024376p' < ttl_tags.csv

    # awk 'NR >= 25024372 && NR <= 25024376' 

    # 28117686-25000000

    # tail -3117686 ttl_tags.csv > ttl_tags2.csv

    # tr < ttl_tags2.csv -d '\000' > ttl_tags3.csv



# * something old

select uniqExact(tag) from tags where weight > 10

# basic selection 
select tag, count(tag) from tags group by tag limit 10

# exclude low loadings
select tag, count(tag) from tags where weight > 10 group by tag limit 10

# exclude unfrequent tags
select tag, count(tag) from tags where weight > 10 group by tag having count(tag) > 5 limit 10

# add average weight 
select tag,
avg(weight) as avg_weight,
count(tag) as cnt,
quantileExact(0.25)(weight) as q25,
quantileExact(0.5)(weight) as q50,
quantileExact(0.75)(weight) as q75
from tags where weight > 10 group by tag having count(tag) > 5 order by avg(weight) desc limit 10 

# need to merge that with playcount
# idk i probably have to make maaaany nested merge statements to get counts of each time period -> can't just use song_info, but have to basically generate a new song_info for each period


## getting number of tags, given restrictions
## nesting statements for DAYS!
select uniqExact(tag) from (select tag, count(*) from tags where weight > 10 group by tag having count(tag) > 10)



