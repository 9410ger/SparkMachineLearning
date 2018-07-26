import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class DiscretizeAttribute{
    public static void main(String args[]) throws Exception{
        //load dataset
        DataSource source = new DataSource("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\lotes.arff");
        Instances dataset = source.getDataSet();
        //set options
        String[] options = new String[5];
        //choose the number of intervals, e.g. 2 :
        options[0] = "-B"; options[1] = "4";
        //choose the range of attributes on which to apply the filter:
        options[2] = "-R";
        options[3] = "1-3";
        options[4] = "-V";
        //Apply discretization:
        Discretize discretize = new Discretize();
        //discretize.setOptions(options);
        discretize.setInputFormat(dataset);
        Instances newData = Filter.useFilter(dataset, discretize);


        ArffSaver saver = new ArffSaver();
        saver.setInstances(newData);
        saver.setFile(new File("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\lotes2.arff"));
        saver.writeBatch();
    }
}