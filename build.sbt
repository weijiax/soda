name := """SODA"""

version := "2.6.x"

scalaVersion := "2.11.11"

lazy val root = (project in file(".")).enablePlugins(PlayScala)

libraryDependencies ++= Seq(
guice,
"com.fasterxml.jackson.core" % "jackson-core" % "2.8.7",
"com.fasterxml.jackson.core" % "jackson-databind" % "2.8.7",
"com.fasterxml.jackson.core" % "jackson-annotations" % "2.8.7",
"com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.8.7",
"org.scalatestplus.play" %% "scalatestplus-play" % "3.1.1" % "test",
"org.apache.spark" % "spark-core_2.11" % "2.1.1",
"org.apache.spark" % "spark-sql_2.11" % "2.1.1",
"org.apache.spark" % "spark-mllib_2.11" % "2.1.1"

)
unmanagedResourceDirectories in Assets += baseDirectory.value / "videos"