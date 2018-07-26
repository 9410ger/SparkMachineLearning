package SparkMachineLearning;


import java.io.PrintWriter;
import java.util.Random;
import java.text.DecimalFormat;

public class LibSvmFileCreator {

    public static void main(String[] args) throws Exception{
        Random random = new Random();
        int temperature;
        int airHumidity;
        int groundHumidity;
        double ligth;
        String ligth2;
        int cropName;
        int sicness;
        DecimalFormat df2 = new DecimalFormat(".##");
        PrintWriter writer = new PrintWriter("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\tizonLibSvm3.txt", "UTF-8");
        for(int i=0; i<250000; i++ ){
            temperature = random.nextInt((25-5)+1)+5;
            airHumidity = random.nextInt((35-15)+1)+15;
            groundHumidity = random.nextInt((35-15)+1)+15;
            ligth = 1.5 + (20.0 - 1.5) * random.nextDouble();
            ligth2 = df2.format(ligth);
            ligth = Double.parseDouble(ligth2.replace(",","."));
            // 1: Papa   2: Tomate   3: Lechuga
            cropName = random.nextInt((3-1)+1)+1;
            if(temperature >= 10 && temperature <= 15 && airHumidity >= 25 && airHumidity <= 30 && groundHumidity >= 22 && groundHumidity <= 29 && ligth >= 3.5 && ligth <= 4.0 && (cropName == 1 || cropName == 2) ){
                sicness = 1;
                writer.println(sicness+" "+"1:"+temperature+" "+"2:"+airHumidity+" "+"3:"+groundHumidity+" "+"4:"+ligth+" "+"5:"+cropName);
            }else{
                sicness = 0;
                writer.println(sicness+" "+"1:"+temperature+" "+"2:"+airHumidity+" "+"3:"+groundHumidity+" "+"4:"+ligth+" "+"5:"+cropName);
            }
        }
        writer.close();
    }
}
