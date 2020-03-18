from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as sf
import re


sc = SparkContext.getOrCreate()
sqlC = SQLContext(sc)
twitter = sqlC.read.json("/var/twitter/decahose/raw/decahose.2019-01*")
filter = twitter.filter((sf.size(twitter.entities.hashtags) > 0) & (sf.size(twitter.entities.media) == 1) & (twitter.lang == 'en'))

# write filter_res to json if we need other data
filter.write.json('OneMonthOrigin')
# filter = sqlC.read.json("OneMonthOrigin")

''' filter text, hashtags, img_url '''
filter = filter.select(filter.text, filter.entities.hashtags.text, filter.entities.media.media_url[0])
filter = filter.toDF('text', 'hashtags', 'image').rdd
# number = 11879798

''' TODO make all hashtags lowercase. METHODS apply function '''

''' filter non english text '''
def isEnglish(s):
    try:
        s.encode('utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

filter = filter.filter(lambda x: isEnglish(x.text))
# number = 5960697

''' filter non english hashtags '''
def filter_hashtag(l):
    tmp = []
    for i in l:
        if isEnglish(i):
            tmp.append(i)
    if tmp == []:
        return None
    return tmp

filter.hashtags = filter.map(lambda x: filter_hashtag(x.hashtags))
filter = filter.toDF().dropna().dropDuplicates().rdd
# filter.toDF().write.json('OneMonthTotal')
# number = 2614503

''' hashtag frequency '''
def hashtag_cnt(l):
    return [(i, 1) for i in l.hashtags]

freq = filter.flatMap(hashtag_cnt).reduceByKey(lambda x,y: x+y).sortBy(lambda (word, count): -count)
# number = 699096

# get top 10,033 hashtags, which means get all hashtag whose freq >= 57
top_freq = dict(freq.top(10033, key=lambda x:x[1]))

def hashtag_isin(l):
    for i in l:
        if i in top_freq:
            return True
    return False

top_freq_twitter = filter.filter(lambda x: hashtag_isin(x.hashtags))
# top_freq_twitter.toDF().write.json('OneMonth10033')
# number 1,741,853