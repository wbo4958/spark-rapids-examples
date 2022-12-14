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

import org.apache.spark.ml.feature.{PCA, VectorAssembler}
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

    val start = System.currentTimeMillis()

    // mean centering via ETL
    val avgValue = df.select(df.schema.names.map(col).map(avg): _*).first()
    val inputCols = (0 until df.schema.names.length).map(i =>
      (col(df.schema.names(i)) - avgValue.getDouble(i)).alias("fea_" + i))
    val meanCenterDf = df.select(inputCols: _*)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(meanCenterDf.columns)
      .setOutputCol("features")

    val dataDf = vectorAssembler.transform(meanCenterDf)

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pca_features")
      .setK(3)

    val pcaModelCpu = pca.fit(dataDf)

    val end = System.currentTimeMillis()


    println("Training done")
    println((end - start) / 1000 + " seconds")

    pcaModelCpu.transform(dataDf).select("pca_features").show(false)
    spark.stop()
  }
}
