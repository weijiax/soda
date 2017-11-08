package controllers

import javax.inject.Inject
import scala.concurrent.{ExecutionContext, Future, Promise}
import org.apache.spark.sql._
import org.apache.spark._
import play.api.i18n.MessagesApi
import play.api.libs.json.Json
import play.api.mvc.{Action, AnyContent, BaseController}

import play.api.mvc._

import play.api.libs.json.Json

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

class SparkController @Inject()(cc: MessagesControllerComponents)
(implicit exec: ExecutionContext) extends MessagesAbstractController(cc) {
   import QueryForm._
   val video_root="public/videos/"
    var spark = SparkSession.builder
      .master("local")
      .appName("ApplicationController")
      .getOrCreate();
    val header = Seq("frame", "obj_id", "isMoving", "car_id", "y_class", "confidence", "x_min", "y_min", "x_max", "y_max", "dx", "dy")
    
    //var ready=false;
    
    def initData(name:String) ={
          val data:DataFrame =spark.read.format("csv").option("inferSchema", "true")
              .load(video_root+name+"/out.log").toDF(header:_*)
          data.createOrReplaceTempView(name)
      //    ready=true;
    }
    //not used
    //def existTable(name: String) : Boolean ={
    //  spark.catalog.tableExists(name)
    //}
  def load (name : String)  = Action {
  //  Action.async{
      
      //sparkSession = Init.getSparkSessionInstance
      val data:DataFrame =spark.read.format("csv").option("inferSchema", "true")
              .load(video_root+name+"/out.log").toDF(header:_*)
      val result : DataFrame = data.describe()
      val rawJson = result.toJSON.collect().mkString
      //Future.successful(Ok(Json.toJson(rawJson)))
      //Ok(views.html.dataframe(name, result))
      Ok(rawJson)
//    }
  }
  
  def runSparkQuery =  Action {
       println("running runSparkQuery")
    
       implicit request: MessagesRequest[AnyContent] => {
         println(request.body.asJson.getOrElse("failed to get request as JSON"))
          val json=request.body.asJson.get
          val table_name= (json \ "table").as[String]
          var qs= "SELECT "+ (json \ "pred").as[String] + " FROM "+(json \ "table").as[String]
          val cond=(json \ "cond").as[String]
          if( ! Option(cond.trim).forall(_.isEmpty))
              qs = qs + " WHERE "+ cond
          println("running : "+ qs)
       if (!spark.catalog.tableExists(table_name)) initData(table_name)
        Ok("["+spark.sql(qs).toJSON.collect().mkString(",")+"]")  
       }
      //Ok(request.body.asText.getOrElse("failed to get request as text"))

  }
  
  def runAnalysis (table_name:String): DataFrame = {
    if (!spark.catalog.tableExists(table_name)) initData(table_name)
    
    val movement = spark.sql("SELECT car_id, min(frame) as start_frame, max(frame) as end_frame from "
          +table_name + " WHERE car_id > 0 GROUP by car_id")
    movement.createOrReplaceTempView("movement")
   
    var df = spark.sql("SELECT movement.car_id, frame, (x_min + x_max) / 2 as x, (y_min + y_max) / 2 as y, array((x_min + x_max) / 2, (y_min + y_max) / 2 ) as xy from "
        +table_name +", movement where movement.car_id = "+table_name+".car_id AND (frame = start_frame OR frame = start_frame)")
    
    val assembler = new VectorAssembler().setInputCols(Array("x", "y")).setOutputCol("features")
    
     runKmeans(assembler.transform(df), 4).orderBy("car_id", "prediction")    
    //return df
  }
  
  def showAnalysis (name: String) = Action{
    //val name="video_1"
    Ok(views.html.dataframe(name, runAnalysis(name)))
  }
  
  def runKmeans(data: DataFrame, k : Int) : DataFrame = {
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val model = kmeans.fit(data)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(data)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println) 
    model.transform(data)
  }
  /*
  def runSparkQuery0 =  Action {
       println("running runSparkQuery")
       implicit request: MessagesRequest[AnyContent] => {
        var query= request.body.asText.getOrElse("select count(*) as total_records")
        if (!ready) initData("video_1")
        Ok(spark.sql(query +" FROM video_1").toJSON.collect().mkString)  
       }
      //Ok(request.body.asText.getOrElse("failed to get request as text"))

  }
  
  
  def index: Action[AnyContent] = { Action.async {
    val query1 =
      s"""
        SELECT * FROM godzilla WHERE date='2000-02-05' limit 8
      """.stripMargin
      val sparkSession = Init. getSparkSessionInstance
      sparkSession.sqlContext.read.csv("conf/data.json")
      val result: DataFrame = sparkSession.sql(query1)
      val rawJson = result.toJSON.collect().mkString
      Future.successful(Ok(Json.toJson(rawJson)))
    }
  }
  */
}