name := "Project1"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.1" % "test" classifier "tests"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.1" % "test" classifier "tests"

libraryDependencies += "io.spray" %%  "spray-json" % "1.3.3"