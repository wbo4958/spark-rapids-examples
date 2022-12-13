/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.nvidia.spark.examples.pca

import org.apache.spark.ml.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]): Unit = {
    println("args v", args.length)
    if (args.length != 1) {
      println("Usage: Main <input>")
      System.exit(1)
    }
    val dataPath = args(0)
    println("Starting PCA example")

    val spark = SparkSession.builder().appName("PCA Example").getOrCreate()
    // load the parquet files
    val df = spark.read.parquet(dataPath)

    val gpuStart = System.currentTimeMillis()
    // mean centering via ETL
    val avgValue = df.select(df.schema.names.map(col).map(avg): _*).first()
    val inputCols = (0 until df.schema.names.length).map(i =>
      (col(df.schema.names(i)) - avgValue.getDouble(i)).alias("fea_" + i))
    val meanCenterDf = df.select(inputCols: _*)

    val dataDf = meanCenterDf.withColumn("feature", array(meanCenterDf.columns.map(col): _*))

    println("GPU training")
    val pcaGpu = new com.nvidia.spark.ml.feature.PCA().setInputCol("feature").setOutputCol("pca_features").setK(3)
    // GPU train

    val pcaModelGpu = pcaGpu.fit(dataDf)
    val gpuEnd = System.currentTimeMillis()
    println("GPU training Done: ")
    println((gpuEnd - gpuStart) / 1000 + " seconds")


    val cpuStart = System.currentTimeMillis()
    // use udf to meet standard CPU ML algo input requirement: Vector input
    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })
    val vectorDf = dataDf.withColumn("feature_vec", convertToVector(col("feature")))
    // use original Spark ML PCA class
    val pcaCpu = new org.apache.spark.ml.feature.PCA().setInputCol("feature_vec").setOutputCol("pca_features").setK(3)
    // CPU train

    val pcaModelCpu = pcaCpu.fit(vectorDf)
    val cpuEnd = System.currentTimeMillis()



    println("CPU training: ")
    println((cpuEnd - cpuStart) / 1000 + " seconds")

    // transform
    pcaModelGpu.transform(vectorDf).select("pca_features").show(false)
    pcaModelCpu.transform(vectorDf).select("pca_features").show(false)
  }
}
