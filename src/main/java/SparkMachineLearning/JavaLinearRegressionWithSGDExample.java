package SparkMachineLearning;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import java.util.Arrays;

public class JavaLinearRegressionWithSGDExample {

    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "C:\\winutils\\");


        SparkConf conf = new SparkConf().setAppName("JavaLinearRegressionWithSGDExample").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);


        // Load and parse the data
        String path = "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\tizon.txt";
        JavaRDD<String> data = sc.textFile(path);
        JavaRDD<LabeledPoint> parsedData = data.map(line -> {
            String[] parts = line.split(" ");
            //String[] features = parts[1].split(" ");
            double[] v = new double[parts.length];
            for (int i = 1; i < parts.length - 1; i++) {
                v[i] = Double.parseDouble(parts[i].split(":")[1]);
            }
            return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
        });
        parsedData.cache();

        // Building the model
        int numIterations = 100;
        double stepSize = 0.00000001;
        //Entrenar el modelo
        LinearRegressionModel model = LinearRegressionWithSGD.train(JavaRDD.toRDD(parsedData), numIterations, stepSize);

        // Evaluate model on training examples and compute training error
        JavaPairRDD<Double, Double> valuesAndPreds = parsedData.mapToPair(point -> new Tuple2<>(model.predict(point.features()), point.label()));

        double MSE = valuesAndPreds.mapToDouble(pair -> {
            double diff = pair._1() - pair._2();
            return diff * diff;
        }).mean();
        System.out.println("training Mean Squared Error = " + MSE);

        // Save and load model
        //model.save(sc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\JavaLinearRegressionWithSGDExample");
        LinearRegressionModel sameModel = LinearRegressionModel.load(sc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\JavaLinearRegressionWithSGDExample");
        double[] vector = {15.0,27.0,25.0,3.7,2.0};
        int[] index = {0,1,2,3,4};
        Vector v = new SparseVector(5,index,vector);
        System.out.println("El modelo predijo que: "+sameModel.predict(v));
        sc.stop();
    }

}
