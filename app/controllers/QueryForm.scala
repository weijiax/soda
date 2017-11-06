package controllers

object QueryForm {
  import play.api.data.Forms._
  import play.api.data.Form

  /**
   * A form processing DTO that maps to the form below.
   *
   * Using a class specifically for form binding reduces the chances
   * of a parameter tampering attack and makes code clearer.
   */
  case class Data(query: String, max_results: Int)
  case class QueryString(pred: String, table: String, cond: String)
  
  val queryForm = Form(
      mapping("pred" ->nonEmptyText,
          "table" -> nonEmptyText,
          "pred" ->nonEmptyText)(QueryString.apply)(QueryString.unapply))

  /**
   * The form definition for the "create a widget" form.
   * It specifies the form fields and their types,
   * as well as how to convert from a Data to form data and vice versa.
   */
  val query = Form(
    mapping(
      "query" -> nonEmptyText,
      "max_results" -> number(max = 200)
    )(Data.apply)(Data.unapply)
  )
}