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
import org.apache.spark.ml.linalg._
import org.apache.spark.sql._
import org.scalatest.FunSuite
import org.apache.spark.Partitioner

abstract class XGBoostClassifierSuiteBase extends FunSuite with PerTest {

  protected val treeMethod: String = "hist"
}

class XGBoostCpuClassifierSuite extends XGBoostClassifierSuiteBase {
  test("XGBoost-Spark XGBoostClassifier output should match XGBoost4j") {
    val trainingDM = new DMatrix(Classification.train.iterator)
    val testDM = new DMatrix(Classification.test.iterator)
    val trainingDF = buildDataFrame(Classification.train)
    val testDF = buildDataFrame(Classification.test)
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF)
  }

  test("XGBoostClassifier should make correct predictions after upstream random sort") {
    val trainingDM = new DMatrix(Classification.train.iterator)
    val testDM = new DMatrix(Classification.test.iterator)
    val trainingDF = buildDataFrameWithRandSort(Classification.train)
    val testDF = buildDataFrameWithRandSort(Classification.test)
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
      "objective" -> "binary:logistic",
      "tree_method" -> treeMethod,
      "max_bin" -> 16)


    val model2 = new XGBoostClassifier(paramMap ++ Array("num_round" -> round,
      "num_workers" -> numWorkers)).fit(trainingDF)

    val prediction2 = model2.transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[DenseVector]("probability"))).toMap

    val model1 = ScalaXGBoost.train(trainingDM, paramMap, round)
    val prediction1 = model1.predict(testDM)

    assert(testDF.count() === prediction2.size)
    // the vector length in probability column is 2 since we have to fit to the evaluator in Spark
    for (i <- prediction1.indices) {
      assert(prediction1(i).length === prediction2(i).values.length - 1)
      for (j <- prediction1(i).indices) {
        assert(prediction1(i)(j) === prediction2(i)(j + 1))
      }
    }

    val prediction3 = model1.predict(testDM, outPutMargin = true)
    val prediction4 = model2.transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[DenseVector]("rawPrediction"))).toMap

    assert(testDF.count() === prediction4.size)
    // the vector length in rawPrediction column is 2 since we have to fit to the evaluator in Spark
    for (i <- prediction3.indices) {
      assert(prediction3(i).length === prediction4(i).values.length - 1)
      for (j <- prediction3(i).indices) {
        assert(prediction3(i)(j) === prediction4(i)(j + 1))
      }
    }

    // check the equality of single instance prediction
    val firstOfDM = testDM.slice(Array(0))
    val firstOfDF = testDF.filter(_.getAs[Int]("id") == 0)
      .head()
      .getAs[Vector]("features")
    val prediction5 = math.round(model1.predict(firstOfDM)(0)(0))
    val prediction6 = model2.predict(firstOfDF)
    assert(prediction5 === prediction6)
  }
}

@GpuTestSuite
class XGBoostGpuClassifierSuite extends XGBoostClassifierSuiteBase {
  override protected val treeMethod: String = "gpu_hist"
  override protected val numWorkers: Int = 1
}
