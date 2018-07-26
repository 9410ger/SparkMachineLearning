package SparkMachineLearning;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.tree.impl.DecisionTreeMetadata;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;

class JavaDecisionTreeClassificationExample {

    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "C:\\winutils\\");

        // Crear la configuración de spark
        SparkConf sparkConf = new SparkConf().setAppName("SparkModeloClasificacion").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        //Cargar el dataset de un archivo libsvm
        String datapath = "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\enfermedadX.txt";
        //Parsear el archivo a RDD
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        //Dividir el dataset 70% para entrenamiento y 30% para pruebas
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Número de respuestas a predecir en este caso (0 ó 1)
        int numClasses = 2;
        //Si el mapa queda vacio se da por entender que todas las caracteristicas o atributos son continuos
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        //Impuridad de nodos de clasificación
        String impurity = "gini";
        //Profundidad del árbol de decisión
        int maxDepth = 5;
        //Número de contenedores utilizados al discretizar las funciones continuas.
        int maxBins = 1000;

        //Entrenar el modelo de clasificación
        DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        // Evalua el modelo entrenado, comparando la predicción con la respuesta real
        JavaPairRDD<Object, Object> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        //Calculo de porcentaje de error
        double testErr = predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testData.count();
        System.out.println("Test Error: " + testErr);
        System.out.println("Learned classification tree model:\n" + model.toDebugString());
        //System.out.println("El modelo predijo que: "+model.predict(v));
        //Guardar el modelo
        model.save(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\SparkModeloClasificacion");
        //Cargar el modelo
        DecisionTreeModel sameModel = DecisionTreeModel.load(jsc.sc(), "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\SparkModeloClasificacion");

        double[] vector = {22.0,40.0,390.0};
        int[] index = {0,1,2};
        Vector v = new SparseVector(3,index,vector);
        System.out.println("El modelo predijo que: "+sameModel.predict(v));
    }
}