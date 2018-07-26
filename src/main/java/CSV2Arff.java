import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class CSV2Arff {

    public static void main(String[] args) throws Exception {

        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\output.csv"));
        System.out.println("Leyó el archivo");
        Instances data = loader.getDataSet();//get instances object

        // save ARFF
        ArffSaver saver = new ArffSaver();
        System.out.println("creó el arff");
        saver.setInstances(data);//set the dataset we want to convert
        //and save as ARFF
        System.out.println("se le colocan las instancias");
        saver.setFile(new File("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\output.arff"));
        saver.writeBatch();
    }
} 