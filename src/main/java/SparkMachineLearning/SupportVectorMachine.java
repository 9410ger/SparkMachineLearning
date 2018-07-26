package SparkMachineLearning;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.util.Arrays;

public class SupportVectorMachine {

    public static void main(String[] args){
        System.setProperty("hadoop.home.dir", "C:\\winutils\\");

        // Crear la configuración de spark
        SparkConf sparkConf = new SparkConf().setAppName("RegresionLogistica").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\enfermedadX.txt";

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();

// Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> training = data.sample(false, 0.7, 11L);
        training.cache();
        JavaRDD<LabeledPoint> test = data.subtract(training);

// Run training algorithm to build the model.
        int numIterations = 100;
        SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        int aciertos = 0;
        float acierto = 0;

        for(LabeledPoint lp: test.collect()){
            System.out.println("Vector de caracteristicas: "+Arrays.toString(lp.features().toArray()));
            System.out.println("Predicción: "+model.predict(lp.features())+" : Respuesta: "+lp.label());
            if(model.predict(lp.features()) == lp.label()){
                aciertos+=1;
            }
        }

        acierto = (aciertos * 100) / test.count();
        System.out.println("El porcentaje de acierto fue: "+acierto);

// Clear the default threshold.
        model.clearThreshold();

// Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

// Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();

        System.out.println("Area under ROC = " + auROC);

// Save and load model
        model.save(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\SupportVectorMachine");
        SVMModel sameModel = SVMModel.load(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\SupportVectorMachine");
    }

}
