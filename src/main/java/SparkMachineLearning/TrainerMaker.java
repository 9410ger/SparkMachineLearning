package SparkMachineLearning;

import java.io.PrintWriter;
import java.util.Random;
import java.text.DecimalFormat;

public class TrainerMaker {

    public static void main(String[] args) throws Exception {
        Random random = new Random();
        double temp;
        double hum;
        double light;
        String temp2;
        String hum2;
        String light2;
        int sickness;
        DecimalFormat df2 = new DecimalFormat(".##");
        PrintWriter writer = new PrintWriter("C:\\Users\\Dell\\Desktop\\NovenoSemestre\\PGR1\\weka\\target\\enfermedadX.txt", "UTF-8");
        for(int i = 0; i < 250000; i++){
            light = 200.0 + (650.0 - 200.0) * random.nextDouble();
            light2 = df2.format(light);
            light = Double.parseDouble(light2.replace(",","."));
            temp = 15.0 + (25.0 - 15.0) * random.nextDouble();
            temp2 = df2.format(temp);
            temp = Double.parseDouble(temp2.replace(",","."));
            hum = 40.0 + (60.0 - 40.0) * random.nextDouble();
            hum2 = df2.format(hum);
            hum = Double.parseDouble(hum2.replace(",","."));

            if((((light/hum)+12) - temp) >= 0.0){
                sickness = 1;
                writer.println(sickness+" 1:"+temp+" 2:"+hum+" 3:"+light);
            }else{
                sickness = 0;
                writer.println(sickness+" 1:"+temp+" 2:"+hum+" 3:"+light);
            }

        }
    }
}
