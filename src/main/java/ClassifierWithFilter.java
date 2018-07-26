import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.filters.unsupervised.attribute.Remove;

public class ClassifierWithFilter{
    public static void main(String args[]) throws Exception{
        //load dataset
        DataSource source = new DataSource("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\lotes.arff");
        Instances dataset = source.getDataSet();
        //set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes()-1);

        //the base classifier
        J48 tree = new J48();
        //the filter
        Remove remove = new Remove();
        //remove.setAttributeIndices("1");
        String[] opts = new String[]{ "-R", "1"};
        //set the filter options
        remove.setOptions(opts);

        //Create the FilteredClassifier object
        FilteredClassifier fc = new FilteredClassifier();
        //specify filter
        //fc.setFilter(remove);
        //specify base classifier
        fc.setClassifier(tree);
        //Build the meta-classifier
        fc.buildClassifier(dataset);

        System.out.println(tree.graph());
    }

}