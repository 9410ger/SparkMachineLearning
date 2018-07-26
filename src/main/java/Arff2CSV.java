

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;
import java.io.File;

public class Arff2CSV {

    public static void main(String[] args) throws Exception {

        // load ARFF
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\airlines.arff"));
        Instances data = loader.getDataSet();//get instances object

        // save CSV
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);//set the dataset we want to convert
        //and save as CSV
        saver.setFile(new File("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\airlines.csv"));
        saver.writeBatch();
    }
}