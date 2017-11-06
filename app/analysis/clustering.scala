package analysis

 import org.apache.spark.sql._
 import org.apache.spark._
 import org.apache.spark.ml.clustering.KMeans
 
class clustering {
  //var spark : SparkSession 

  
  def runKmeans(data: DataFrame, k : Int) {
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val model = kmeans.fit(data)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(data)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println) 
  }
}