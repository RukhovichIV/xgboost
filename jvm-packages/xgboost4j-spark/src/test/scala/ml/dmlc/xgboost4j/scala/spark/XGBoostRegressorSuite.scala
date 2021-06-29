/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.java.GpuTestSuite
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.scalatest.FunSuite

abstract class XGBoostRegressorSuiteBase extends FunSuite with PerTest {
  protected val treeMethod: String = "hist"

  test("XGBoost-Spark XGBoostRegressor output should match XGBoost4j") {
    val trainingDM = new DMatrix(Regression.train.iterator)
    val testDM = new DMatrix(Regression.test.iterator)
    val trainingDF = buildDataFrame(Regression.train)
    val testDF = buildDataFrame(Regression.test)
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF)
  }

  test("XGBoostRegressor should make correct predictions after upstream random sort") {
    val trainingDM = new DMatrix(Regression.train.iterator)
    val testDM = new DMatrix(Regression.test.iterator)
    val trainingDF = buildDataFrameWithRandSort(Regression.train)
    val testDF = buildDataFrameWithRandSort(Regression.test)
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF)
  }

  private def checkResultsWithXGBoost4j(
      trainingDM: DMatrix,
      testDM: DMatrix,
      trainingDF: DataFrame,
      testDF: DataFrame,
      round: Int = 5): Unit = {
    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "silent" -> "1",
      "objective" -> "reg:squarederror",
      "max_bin" -> 16,
      "tree_method" -> treeMethod)

    val model1 = ScalaXGBoost.train(trainingDM, paramMap, round)
    val prediction1 = model1.predict(testDM)

    val model2 = new XGBoostRegressor(paramMap ++ Array("num_round" -> round,
      "num_workers" -> numWorkers)).fit(trainingDF)

    val prediction2 = model2.transform(testDF).
        collect().map(row => (row.getAs[Int]("id"), row.getAs[Double]("prediction"))).toMap

    assert(prediction1.indices.count { i =>
      math.abs(prediction1(i)(0) - prediction2(i)) > 0.01
    } < prediction1.length * 0.1)


    // check the equality of single instance prediction
    val firstOfDM = testDM.slice(Array(0))
    val firstOfDF = testDF.filter(_.getAs[Int]("id") == 0)
        .head()
        .getAs[Vector]("features")
    val prediction3 = model1.predict(firstOfDM)(0)(0)
    val prediction4 = model2.predict(firstOfDF)
    assert(math.abs(prediction3 - prediction4) <= 0.01f)
  }
}

class XGBoostCpuRegressorSuite extends XGBoostRegressorSuiteBase {

}

@GpuTestSuite
class XGBoostGpuRegressorSuite extends XGBoostRegressorSuiteBase {
  override protected val treeMethod: String = "gpu_hist"
  override protected val numWorkers: Int = 1
}
