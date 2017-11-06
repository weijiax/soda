package controllers

import javax.inject.Inject

import play.api.mvc._
import play.api.data._
import play.api.i18n._

import play.api.libs.json._
import java.io.FileInputStream
import java.io.File


class HomeController @Inject()(cc: MessagesControllerComponents) extends MessagesAbstractController(cc) {
 import QueryForm._
 import scala.concurrent.ExecutionContext.Implicits.global
  val video_root="public/videos/"
   def index = Action {
    val d = new File(video_root)
    if (d.exists && d.isDirectory) {
        Ok(views.html.index(d.listFiles.filter(_.isDirectory).map( file => file.getName).toList))
    } else {
        Ok(views.html.index(List[String]()))
    }
  }
 
 def getVideo(path:String, name:String) = Action{
   println("get file: "+ name +" at Path: "+path)
   Ok.sendFile(new File(path + name.replace("%20", " ")))
 }
  
 
 
 
 
 
   
 def test = Action {
   println("in test function")
    Ok(views.html.test())
  }
 
  def searchVideo (videoName:String) = Action{
    println("running SearchVideo")
    implicit request: MessagesRequest[AnyContent] =>
    Ok(views.html.searchVideo(videoName, query))
  }
  
  def showMetaInfo (name :String)  = Action{
    val stream = new FileInputStream(video_root+name+"/metadata.json")
    val json = try {  Json.parse(stream) } finally { stream.close() }
    Ok(json)
  }

  // This will be the action that handles our form post
    // This will be the action that handles our form post
  def runQuery = Action { 
    println("running runQuery")
  request: MessagesRequest[AnyContent] =>
      Ok(request.body.asText.getOrElse("failed to get request as text"))
      
    
  }
  
  def runQuery0 (name : String) = Action { 
    println("running runQuery0")
    implicit request: MessagesRequest[AnyContent] =>
    
    val errorFunction = { formWithErrors: Form[Data] =>
      // This is the bad case, where the form had validation errors.
      // Let's show the user the form again, with the errors highlighted.
      // Note how we pass the form with errors to the template.
    print("error function")
      BadRequest(views.html.searchVideo(name, formWithErrors))
    }

    val successFunction = { data: Data =>
      // This is the good case, where the form was successfully parsed as a Data.
      //val widget = Widget(name = data.name, price = data.price)
      //widgets.append(widget)
      //Redirect(routes.WidgetController.listWidgets()).flashing("info" -> "Widget added!")
       print("success function")
       printf("Query is %s with maximum %d results expected", data.query, data.max_results)
       //Redirect(routes.HomeController.searchVideo(name)).flashing("query" -> data.query)
       Ok(data.query)
    }
    print("runQuery0")
    val formValidationResult = query.bindFromRequest
       formValidationResult.fold(errorFunction, successFunction)
    
  }
}