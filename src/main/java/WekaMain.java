

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.*;

public class WekaMain {

    public static void main(String[] args) throws Exception{
        Instances dataset = new Instances(new BufferedReader(new FileReader("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\airlines.arff")));
        System.out.println(dataset.toSummaryString());
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File("new.arff"));
        saver.writeBatch();
    }

}
