/**Implementation of Project 1
 *
 * Work done by Ankita Joshi
 *
 */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import nb.NaiveBayes


object nbayes {

    def main(args: Array[String]): Unit ={
    
        //Create the sparkcontext
        val conf = new SparkConf().setAppName("project1")
        val sc = new SparkContext(conf)

        //Path to where all train and test files are located input as command-line argument
        val trainX = args(0)
        val trainy = args(1)
        val testX = args(2)
        val testy = args(3)
        val stopwordsPath = args(4)

        //Read the train and test data
        val X = sc.textFile(trainX).zipWithIndex.map{case(k,v) => (v,k)}
        val y = sc.textFile(trainy).zipWithIndex.map{case(k,v) => (v,k)}
        val tX = sc.textFile(testX).zipWithIndex.map{case(k,v) => (v,k)}
        val ty = sc.textFile(testy).zipWithIndex.map{case(k,v) => (v,k)}
        val stopwords = sc.broadcast(sc.textFile(stopwordsPath).collect())

        //Combine the docs with their labels
        val documentsWithLabels = X.join(y)
        //documentsWithLabels.foreach(println(_))
        val data = documentsWithLabels.map{case(stuff) => 
            val doc = stuff._2._1
            val l = stuff._2._2.split(",")
            l.map{case(a) =>  (a,doc)}         
            }.flatMap(x => x).filter{ case(a,doc) => a contains "CAT"} 
      
        val totalDocuments = data.count()   

        //Train
        val (model,vocabulary) = NaiveBayes.train(data,totalDocuments,stopwords)

        //Test
        //Broadcast the vocabulary
        val vo = sc.broadcast(vocabulary.toInt)
        val result = NaiveBayes.test(tX,ty,model,vo,stopwords)  

    }
 
}
