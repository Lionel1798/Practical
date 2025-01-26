# Practical

****Aim: Filtering RDDs, and the Minimum Temperature by Location**

from pyspark import SparkConf, SparkContext

conf=SparkConf().setMaster("local").setAppName("MinTemperatures")
sc = SparkContext(conf=conf)

def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0
    return (stationID, entryType, temperature)

lines = sc.textFile("file:///SparkCourse/1800.csv")
parsedLines = lines.map(parseLine)

minTemps = parsedLines.filter(lambda x: "TMIN" in x[1])
stationTemps = minTemps.map(lambda x: (x[0], x[2]))
minTemps = stationTemps.reduceByKey(lambda x, y: min(x, y))
results = minTemps.collect()

for result in results:
  print(result[0] + "\t{:.2f}F".format(result[1]))

  
**#Aim: Counting Word Occurrences using flatMap()#**

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf=conf)

input = sc.textFile("book.txt")
word = input.flatMap(lambda x: x.split())
wordCounts = word.countByValue()

print(wordCounts)

for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if cleanWord:
        print(cleanWord.decode() + " " + str(count))


**##Aim:Executing SQL commandsand SQL-style functionsona Data Frame.##**

from pyspark.sql import SparkSession, Row
import collections

spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("SparkSQL").getOrCreate()

def mapper(line):
    fields = line.split(',')
    return Row(ID=int(fields[0]), name=str(fields[1].encode("utf-8")), age=int(fields[2]), numFriends=int(fields[3]))

lines = spark.sparkContext.textFile("fakefriends.csv")

people = lines.map(mapper)	

schemaPeople = spark.createDataFrame(people).cache()
schemaPeople.createOrReplaceTempView("people")

teenagers = spark.sql("SELECT * FROM people WHERE age >= 13 AND age <= 19")

for teen in teenagers.collect():
    print(teen)

schemaPeople.groupBy("age").count().orderBy("age").show()

spark.stop()


**##Aim:Implement Total Spent by Customer with Data Frames.##**

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("SpendByCustomer")
sc = SparkContext(conf=conf)

def extractCustomerPricePairs(line):
    fields = line.split(',')
    return (int(fields[0]), float(fields[2]))

input = sc.textFile("customer-orders.csv")
mappedInput = input.map(extractCustomerPricePairs)
totalByCustomer = mappedInput.reduceByKey(lambda x, y: x + y)
results = totalByCustomer.collect()

for result in results:
    print(result)


**##Aim:UseBroadcastVariablestoDisplayMovieNamesInsteadofID Numbers##**

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions

def extractValue(line):
  fields = line.split(',')
  return (int(fields[0],int(fields[2])))

def extractValue(line):
  fields = line.split(',')
  # Convert fields[2] to float first, then to int if necessary
  return (int(fields[0]), int(float(fields[2])))

lines = sc.textFile("customer-orders.csv")
mappedInput = lines.map(extractValue)
totalTimeSpend = mappedInput.reduceByKey(lambda x, y :x + y)
results = totalTimeSpend.collect()
for result in results :
  print(result)



**##Aim:Create Similar Movies from One Million Rating##**

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName('recommendation').getOrCreate()

data = spark.read.csv('ratings1.csv', inferSchema=True, header=True)
data.head()
data.printSchema()
data.describe().show()

(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(train_data)

predictions = model.transform(test_data)
predictions.show()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

single_user = test_data.filter(test_data['userId'] == 12).select(['movieId', 'userId'])
single_user.show()

recommendations = model.transform(single_user)
recommendations.orderBy('prediction', ascending=False).show()
