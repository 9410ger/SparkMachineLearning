
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

public class EvaluationWeka {

    public static void main(String[] args)throws Exception {
        String[] options = new String[2];
        options[0] = "-t";
        options[1] = "C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\airlines.arff";
        System.out.println(Evaluation.evaluateModel(new J48(), options));
    }
}
