

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SaveLoadModel {
    public static void main(String args[]) throws Exception{

        DataSource source = new DataSource("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\lotes.arff");
        // 80% de los datos del archivo para entrenamiento
        int trainSize = (int) Math.round(source.getDataSet().numInstances() * 0.8);

        //Se crea la instancia del 80% de los datos
        Instances trainDataset = new Instances(source.getDataSet(), 0, trainSize);
        //Se setea el valor que se quiere predecir para el conjunto de entrenamiento
        trainDataset.setClassIndex(source.getDataSet().numAttributes() - 1);
        //Saber cuantas clases hay por atributo del conjunto de entrenamiento
        int numClasses = trainDataset.numClasses();
        //Imprime un número en orden y luego la clase correspondiente del conjunto de entrenamiento
        for(int i = 0; i < numClasses; i++){
            //get class string value using the class index
            String classValue = trainDataset.classAttribute().value(i);
            System.out.println("Class Value "+i+" is " + classValue);
        }
        //Crear y construir el clasificador
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainDataset);
		//save model
		weka.core.SerializationHelper.write("modelo_entrenado_lotes.model", nb);


        //Cargar modelo
        NaiveBayes nb2 = (NaiveBayes) weka.core.SerializationHelper.read("modelo_entrenado_lotes.model");
        // 20% restante para las pruebas
        int testSize = source.getDataSet().numInstances() - trainSize;
       //Crear la instancia de pruebas con el 20% restante de datos
        Instances testDataset = new Instances(source.getDataSet(), trainSize, testSize);
		//Se setea el valor que se quiere predecir para el conjunto de pruebas
		testDataset.setClassIndex(source.getDataSet().numAttributes() - 1);

        System.out.println("===================");
        System.out.println("Actual Class, NB Predicted");
        for (int i = 0; i < testDataset.numInstances(); i++) {
            //get class double value for current instance
            double actualClass = testDataset.instance(i).classValue();
            //get class string value using the class index using the class's int value
            String actual = testDataset.classAttribute().value((int) actualClass);
            //get Instance object of current instance
            Instance newInst = testDataset.instance(i);
            //call classifyInstance, which returns a double value for the class
            double predNB = nb2.classifyInstance(newInst);
            //use this value to get string value of the predicted class
            String predString = testDataset.classAttribute().value((int) predNB);
            System.out.println(actual+", "+predString);
        }


    }

}
