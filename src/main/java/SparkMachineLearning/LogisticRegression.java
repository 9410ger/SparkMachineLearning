package SparkMachineLearning;

import org.apache.spark.SparkConf;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import scala.Array;
import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.util.Arrays;

public class LogisticRegression {

    public static void main(String[] args){

        System.setProperty("hadoop.home.dir", "C:\\winutils\\");

        // Crear la configuración de spark
        SparkConf sparkConf = new SparkConf().setAppName("RegresionLogistica").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\enfermedadX.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();

// Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits =
                data.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

// Run training algorithm to build the model.
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(2)
                .run(training.rdd());

// Clear the prediction threshold so the model will return probabilities
        /*double[] vector = {15.0,25.0,25.0,3.7,2.0};
        int[] index = {0,1,2,3,4};
        Object t = model.getThreshold().get();
        Vector v = new SparseVector(5,index,vector);
        System.out.println("Prediccion: "+model.predict(v));
        model.clearThreshold();
        System.out.println("Probabilidad: "+model.predict(v));
        model.setThreshold((Double) t);
        System.out.println("Prediccion: "+model.predict(v));*/



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

/*// Compute raw scores on the test set.
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

// Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabels.rdd());

// Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());

// Recall by threshold
        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());

// F Score by threshold
        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.collect());

        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.collect());

// Precision-recall curve
        JavaRDD<?> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.collect());

// Thresholds
        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

// ROC Curve
        JavaRDD<?> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect());

// AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

// AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());*/

// Save and load model
        model.save(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\RegresionLogistica");
        LogisticRegressionModel.load(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\RegresionLogistica");
    }

}
