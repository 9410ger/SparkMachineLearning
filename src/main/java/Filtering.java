import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.FileReader;

public class Filtering {

    public static void main(String[] args) throws Exception{
            Instances dataset = new Instances(new BufferedReader(new FileReader("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\lotes.arff")));
            dataset.setClassIndex(dataset.numAttributes() - 1);
            int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
            int testSize = dataset.numInstances() - trainSize;
            Instances train = new Instances(dataset, 0, trainSize);
            Instances test = new Instances(dataset, trainSize, testSize);
            // filter/ remove 1st attribute
            // classifier
            J48 j48 = new J48();
            j48.setUnpruned(true);        // using an unpruned J48
            // meta-classifier
            FilteredClassifier fc = new FilteredClassifier();
            fc.setClassifier(j48);
            // train and make predictions
            fc.buildClassifier(train);
            for (int i = 0; i < test.numInstances(); i++) {
                double pred = fc.classifyInstance(test.instance(i));
                System.out.print("ID: " + test.instance(i).value(0));
                System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
                System.out.println(", predicted: " + test.classAttribute().value((int) pred));
        }
    }
}
