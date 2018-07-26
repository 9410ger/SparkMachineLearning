
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.io.BufferedReader;
import java.io.FileReader;


public class TrainTestSet {

    public static void main(String[] args) throws Exception{
        Instances dataset = new Instances(new BufferedReader(new FileReader("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\lotes2.arff")));
        dataset.setClassIndex(dataset.numAttributes() - 1);
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        Instances train = new Instances(dataset, 0, trainSize);
        Instances test = new Instances(dataset, trainSize, testSize);
        // train classifier
        Classifier cls = new J48();
        cls.buildClassifier(train);
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));

    }
}
