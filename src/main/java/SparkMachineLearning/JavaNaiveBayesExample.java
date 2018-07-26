package SparkMachineLearning;

import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;

import java.util.Arrays;

public class JavaNaiveBayesExample {

    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "C:\\winutils\\");

        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        String path = "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\tizon.txt";

        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();

        JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.7, 0.3});

        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();

        for(LabeledPoint lp: test.collect()){
            System.out.println("Vector de caracteristicas: "+Arrays.toString(lp.features().toArray()));
            System.out.println("Predicción: "+model.predict(lp.features())+" : Respuesta: "+lp.label());
        }

        System.out.println("Accuracy: "+accuracy);
        System.out.println("Model Bayes format version: "+model.formatVersion());


        // Save and load model
        //model.save(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\JavaNaiveBayesExample");
        NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\JavaNaiveBayesExample");
        double[] vector = {15.0,27.0,18.0,3.7,2.0};
        int[] index = {0,1,2,3,4};
        Vector v = new SparseVector(5,index,vector);
        System.out.println("El modelo predijo que: "+sameModel.predict(v));
        jsc.stop();
    }



}
