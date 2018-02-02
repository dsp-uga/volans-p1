 /**Implementation of Project 1
 *
 * Work done by Ankita Joshi
 *
 */

package nb

import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext
import java.io._

object NaiveBayes {

	def train(data: RDD[(String, String)], totalDocuments: Long, B: Broadcast[Array[String]]): (RDD[(String,String,Double,Double)],Long) ={
		
		
		val sc = data.sparkContext
		//Collect the braodcasted stopwords
		val stopwords = B.value.toArray
		
		//Step 1: Find P(Cj) = Number of documents of label j/ Total nnumber of documents
		val PCj = data.map{case(label,doc) => (label,1)}.reduceByKey(_+_).map{case(label,count) => (label,math.log((count.toFloat/totalDocuments.toFloat)))}
		//PCj.foreach(println(_))

		/*Step 2: Compute P(wi|Label) = count of "word" in label + 1 / [total number of words in label] + |v|
			Denominator is redundant, so just compute it once.
			Maintain the label as the key
			Create a single "mega document" with label,words in that label from all the documents for that label.
			Of course also preprocess the words.
		*/

		//Take the RDD and separate out each word with its label. Ouput: (label,word)
		val wordsOfDoc =  data.flatMap{ case(label,doc) => 
			val words = doc.split("\\s") 
			words.map{case(a) => (label,a)
			}
		}
		//wordsOfDoc.foreach(println(_))

		//Preprocess the above result to count the unique words for each label. Output: ((key,word), count_for_that_word)
		val vv = wordsOfDoc.map{case(doc,word) => 
			val clean = word.toLowerCase.replaceAll("&amp;", "").replaceAll("&quot;","").replaceAll("""([\p{Punct}]|\b\p{IsLetter}{1,2}\b)\s*""", "")
			(doc,clean)
		}.filter{case(doc,word) => word.length >1 && !stopwords.contains(word) }.map{case(doc,word) => ((doc,word),1)}.reduceByKey{case(accCount,count) => accCount + count}
		//vv.foreach(println(_))
		//&& !word.exists(_.isDigit)

		//Compute the Vocabulary: Unique number of words in all documents combined. (Needed for laplace smoothing)
		//Broadcast the vocabulary
		val vocabulary =  vv.map{case((label,word),count) => word}.distinct()
		val vocabCount = vocabulary.count
		val v = sc.broadcast(vocabulary.collect())

		//Compute total number of words for each label(Denominator of step 2)
		val denominators = vv.map{case((label,word),count) => (label,count)}.reduceByKey{case(accCount,count) => accCount + count}.map{case(label,count)=> (label,(count+vocabCount).toFloat)}

		val step1 = vv.map{case((label,word),count) => (label,(word,count))}.groupByKey()

		//Add words not present in a given label, but present in others.
		val addwords = step1.map{case(label,list) => 
			var l= list.toMap
			
			val allwords = v.value.toArray
			for(i <- allwords){
				if(!l.contains(i)){
					val addition = (i,0)
					l+=addition
				}

			}
			val lis = l.toArray

			lis.map{case(a,c) => (label,(a,c))}
			
		}.flatMap(x => x)

		//Now we need to find the probability, Output: Numerator/Denominator. Join both on the label.
		val joined = addwords.join(denominators)

		val Pwi_cj = joined.map{case(label,((word,count),denom)) => (label,(word,math.log((count.toFloat+1.0)/denom)))}

		//Return the trained model, combining the priors and the conditional probabilities.
		val model = PCj.join(Pwi_cj).map{case(label, ((prob1, (word, prob2)))) => (word,label,prob1,prob2)}
		(model,vocabCount.toInt)
	}
	


	def test(data: RDD[(Long, String)], labels: RDD[(Long,String)], model: RDD[(String,String,Double,Double)], v: Broadcast[Int],B: Broadcast[Array[String]]): Unit ={	
		
		//Collect the braodcasted stopwords
		val stopwords = B.value.toArray

		//Create the test data, preprocessing
		val testdata =  data.flatMap{ case(index,doc) => 
			val words = doc.split("\\s") 
			words.map{case(a) => 
				(a.toLowerCase.replaceAll("&amp;", "").replaceAll("&quot;","").replaceAll("""([\p{Punct}]|\b\p{IsLetter}{1,2}\b)\s*""", ""),index)

				
			}.filter{case(word,index) => word.length >1 && !stopwords.contains(word)}
		}
		//&& !word.exists(_.isDigit
		//testdata.foreach(println(_))		

		//Get the priors from training model
		val priors = model.map{case(word,label,pj,pwicj) => (label,pj)}.distinct()
		//get the conditional probabilities from the training model
		val pjs = model.map{case(word,label,pj,pwicj) => (word,(label,pwicj))}
		
		//Find the conditional probs for the test data words
		val condProb = testdata.leftOuterJoin(pjs)
		val condProb1 = condProb.filter{case(word,(index,stuff)) => stuff!= None}
		val v1 = condProb1.map{case(word,(index,stuff)) =>
			
				val t = stuff.get
				((t._1,index),t._2)
				
			
		}/*.reduceByKey{case(accCount,count) => accCount + count }.map{case((label,index),score) => (label,(index,score))}.join(priors).map{case((label,((index,val1),val2)))=>
			(index,(val1.toFloat+val2.toFloat,label))
			}.groupByKey().map{case(doc,list) => (doc,list.toArray.sortWith(_._1 > _._1)(0))}.sortByKey(ascending=true)*/

		

		val condProb2 = condProb.filter{case(word,(index,stuff)) => stuff == None}
		val v2 = condProb2.map{case(word,(index,stuff)) =>

			val vocab = v.value
			val arr = Array("GCAT","MCAT","ECAT","CCAT")
			//println(vocab)
			val smoothing = math.log(1/(vocab.toFloat))

			arr.map{case(a) => ((a,index),smoothing)}//.flatMap(x=> x)

				
				
		}.flatMap{x => x}

		//Combine and fin the highest for each label.
		val combined = v1.union(v2).reduceByKey{case(accCount,count) => accCount + count }.map{case((label,index),score) => (label,(index,score))}.join(priors).map{case((label,((index,val1),val2)))=>
			(index,(val1.toFloat+val2.toFloat,label))
			}.groupByKey().map{case(doc,list) => (doc,list.toArray.sortWith(_._1 > _._1)(0))}.sortByKey(ascending=true)


		val k = combined.sortBy(_._1,ascending = true)//map{ case(docid,(socre,prediction)) => prediction}.saveAsTextFile("/home/ankita/vyom/small/result.txt")
		//val k = v1.sortBy(_._1,ascending = true)

		val result = k.collect()

		//Wrtiting results to the output file
		val writer = new PrintWriter(new File("results.txt"))
    	for((k,v) <- result){

      		writer.write(v._2.toString+"\n")

    	}
		writer.close()

	}
	
}

