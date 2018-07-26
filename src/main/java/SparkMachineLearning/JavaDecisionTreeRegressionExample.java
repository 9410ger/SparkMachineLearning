package SparkMachineLearning;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;

public class JavaDecisionTreeRegressionExample {

    public static void main(String[] args){

        System.setProperty("hadoop.home.dir", "C:\\winutils\\");
        SparkConf sparkConf = new SparkConf().setAppName("SparkModeloRegresion").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        // Load and parse the data file.
        String datapath = "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\tizon.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Set parameters.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        // Impuridad de nodos de regresión
        String impurity = "variance";
        int maxDepth = 5;
        int maxBins = 32;

        // Train a DecisionTree model.
        DecisionTreeModel model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        for(LabeledPoint lp: testData.collect()){
            System.out.println("Vector de caracteristicas: "+Arrays.toString(lp.features().toArray()));
            System.out.println("Predicción: "+model.predict(lp.features())+" : Respuesta: "+lp.label());
        }

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double testMSE = predictionAndLabel.mapToDouble(pl -> { double diff = pl._1() - pl._2();
        return diff * diff;
        }).mean();

        System.out.println("Test Mean Squared Error: " + testMSE);
        System.out.println("Learned regression tree model:\n" + model.toDebugString());

        // Save and load model
        model.save(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\SparkModeloRegresion");
        DecisionTreeModel sameModel = DecisionTreeModel.load(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\SparkModeloRegresion");

        double[] vector = {15.0,27.0,18.0,3.7,2.0};
        int[] index = {0,1,2,3,4};
        Vector v = new SparseVector(5,index,vector);
        System.out.println("El modelo predijo que: "+sameModel.predict(v));

    }
}
